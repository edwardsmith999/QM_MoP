import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import sys

from ase import units
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.io import read, write, vasp

from MOP import get_MOP_kinetic, get_MOP_stress_power, bin_MD
from MOP import reconstruct_and_plot
from MOP import reconstruct_energy_flux_ref
from utils import printenergy,  get_atom_potential_energies, check_torque_conservation

#Define all simulation parameters here
AtomDict = {"Zr":40, "O":8, "H":1}

nsteps = 100
Nbins = 400
savefreq = 1
Nevery = 1

timing = False
checks = False
read_vasp = False

outdir = "./results/"
modelpath = "./foundation_models/"
intialfiles = "./initial_states/"

dynamics = "leapfrog" # "verlet" or "leapfrog"
maceversion = "system" #system or "custom"
fijtype = "dUdrij"   # "dUidrj" or "dUdrij", note dUidrj is very slow

Tset = 500 #Temperature
dt = 0.5 * units.fs

if maceversion == "custom":
    #This should be version from https://github.com/edwardsmith999/mace
    #currently, which adapts and adds fij support
    #This version is adapted from MACE release 0.3.11
    #and seems to require the following to work with acceleration
    #cuequivariance-torch==0.3.0 and e3nn==0.4.4
    import os
    if not os.path.isdir('MACE'):
        try:
            print("Attempting to clone custom MACE version from GitHub edwardsmith999/mace")
            from git import Repo
            Repo.clone_from("https://github.com/edwardsmith999/mace", "MACE")
        except ImportError:
            raise ImportError("Download failed - need custom version of MACE from edwardsmith999")

    #This hacky local file import works well as it prevents changing system mace 
    sys.path.insert(1, os.path.abspath("./MACE"))
    import mace
    assert mace.__file__ == os.path.abspath("./MACE") + "/mace/__init__.py"
    from mace.calculators import MACECalculator
    calc = MACECalculator(modelpath+"mace-mpa-0-medium.model", 
                          device='cuda', 
                          enable_cueq=True,          #These only work with cuequivariance-torch==0.3.0 and e3nn==0.4.4
                          compile_mode="default")   
    tols = 1e-8 #Runs at 64 bit so more accurate

elif maceversion == "system":

    #We require later than MACE version 0.3.13 for stresses, otherwise custom version used
    import mace
    assert (int(mace.__version__.split(".")[2]) >= 13 and 
            int(mace.__version__.split(".")[1]) >= 3 and
            int(mace.__version__.split(".")[0]) >= 0)
    #Should be able to use installed system MACE here
    from mace.calculators import mace_mp
    calc = mace_mp(model=modelpath+"mace-mpa-0-medium.model",
                   device="cuda",
                   compute_atomic_stresses=True, 
                   compute_edge_forces=True,
                   default_dtype="float32")
    tols = 1e-2

#Start from Yang et al VASP file or ASE format initial state
#which has already been equilibrated with velocities applied
if read_vasp:
    atoms = vasp.read_vasp(intialfiles+"water_ZrO2.vasp")
    MaxwellBoltzmannDistribution(atoms, temperature_K=Tset)
    #Remove drift velocity
    Stationary(atoms)
else:
    atoms = read(intialfiles+"water_ZrO2.traj")

#Get system sizes
N = len(atoms)
cell = atoms.cell
pbc = atoms.pbc
Lz = cell[2][2]
binrange = np.linspace(0, Lz, Nbins)

#Set atom to mace calculator
atoms.calc = calc

#Dynamics
if dynamics == "verlet":
    dyn = VelocityVerlet(atoms, dt)
elif "leapfrog":
    #We'll do this explicitly instead
    dyn = None

# Write initial data
atoms.write(outdir+'mace_run.xyz', append=False)
printenergy(atoms, 0)

MOPstress_c_hist = []
MOPstress_k_hist = []
mv_MOP_hist = []

MOPenergy_c_hist = []
MOPenergy_k_hist = []
energy_MOP_hist = []

Pcbins = []
Pkbins = []

Fdotv = np.zeros(nsteps)
dE_dt = np.zeros(nsteps)

#Get atom properties
atomtype = atoms.get_chemical_symbols()
atomno = atoms.get_atomic_numbers()
m = atoms.get_masses()
r = atoms.get_positions()
mv = atoms.get_momenta()
v = np.array([mv[:,i]/m for i in range(3)]).T
KE = 0.5*m*np.sum(v**2, axis=1)
PE = get_atom_potential_energies(atoms)
E = KE + PE

#Now run the dynamics
for t in range(nsteps):

    #Save before next step
    if t % savefreq == 0:
        #print("Writing backup at step", t)
        try:
            atoms.write(outdir+'mace_run.xyz', append=True)
        except ValueError:
            pass
        np.save(outdir+f"MOPstress_c_hist.npy", np.array(MOPstress_c_hist))
        np.save(outdir+f"MOPstress_k_hist.npy", np.array(MOPstress_k_hist))
        np.save(outdir+f"Pcbins.npy", np.array(Pcbins))
        np.save(outdir+f"Pkbins.npy", np.array(Pkbins))
        np.save(outdir+f"mv_MOP_hist.npy", np.array(mv_MOP_hist))
        write(outdir+"water_ZrO2_MOP_checkpoint{:05}.traj".format(t), atoms)   

        np.save(outdir+f"MOPenergy_c_hist.npy", np.array(MOPenergy_c_hist))
        np.save(outdir+f"MOPenergy_k_hist.npy", np.array(MOPenergy_k_hist))
        np.save(outdir+f"energy_MOP_hist.npy", np.array(energy_MOP_hist))

    if timing:
        t0 = time.time()

    #Replace dyn with this
    if dynamics == "verlet":
        r_prev = r.copy()  # Save previous positions before updating
        mv_prev = mv.copy() # Save previous momentum
        E_prev = E.copy() # Save previous energy
        dyn.run(1)
    elif "leapfrog":
        #Time integration
        E_prev = E.copy() # Save previous energy
        f = atoms.get_forces()
        Fdotv[t] = np.sum(np.einsum('ij,ij->i', f, v))
        mv_prev = mv.copy() # Save previous momentum
        mv += dt * f

        atoms.set_momenta(mv)
        r_prev = r.copy() # Save previous positions before updating
        r[:,0] += dt * mv[:,0]/m[:]
        r[:,1] += dt * mv[:,1]/m[:]
        r[:,2] += dt * mv[:,2]/m[:]
        atoms.set_positions(r)

    #Skip unless Nevery
    if t % Nevery != 0:
        continue

    if timing:
        t1 = time.time()
        print("Force update time=", t1-t0)

    #Get atom properties
    atomtype = atoms.get_chemical_symbols()
    atomno = atoms.get_atomic_numbers()
    m = atoms.get_masses()
    r = atoms.get_positions()
    mv = atoms.get_momenta()
    v = np.array([mv[:,i]/m for i in range(3)]).T

    #Get v at half timestep for power calculation
    f = atoms.get_forces()
    mv_next = mv + 0.5 * dt * f
    v_next = np.array([mv_next[:,i]/m for i in range(3)]).T

    #Remove drift velocity
    #Stationary(atoms)
    #print("step=", t, "Drift v=", np.sum(mv,0), np.sum(atoms.get_momenta(),0))

    #Kinetic Energy/Temperature
    KE = 0.5*m*np.sum(v**2, axis=1)
    PE = get_atom_potential_energies(atoms)
    E = KE + PE
    dE_dt[t] = np.sum((E - E_prev) / dt)

    #Check sum of local temperature adds to total
    assert abs(atoms.get_temperature() - 2.*np.sum(KE) / (3 * N * units.kB)) < 1e-5
    assert abs(atoms.get_potential_energy() - np.sum(PE)) < tols

    #With adapted MACE, we get fij force (note atoms.calc.mixer.calcs[0] 
    #if using dispersion but this will fail to provide force balance)
    if fijtype == "dUdrij":
        if maceversion == "custom":
            fij = 2.0*atoms.calc.results["fij"]
            fij[:,:,0] = 0.5*(fij[:,:,0] - fij[:,:,0].T)
            fij[:,:,1] = 0.5*(fij[:,:,1] - fij[:,:,1].T)
            fij[:,:,2] = 0.5*(fij[:,:,2] - fij[:,:,2].T)

        #This version work with  MACE 0.3.14 as it gets fij from edge_forces
        #however it uses hidden functions like "calc._atoms_to_batch"
        #which might change in updates to MACE. 
        elif maceversion == "system":

            #A low level call to get force per atom
            model = atoms.calc.models[0]
            batch_base = atoms.calc._atoms_to_batch(atoms)
            batch = atoms.calc._clone_batch(batch_base)
            out = model(batch.to_dict(), compute_stress=True, compute_edge_forces=True, training=True)
            grad_rij = out["edge_forces"]
            dense = torch.zeros((N, N, grad_rij.shape[1]), device=grad_rij.device, dtype=grad_rij.dtype)
            sender, receiver = batch["edge_index"]
            dense[sender, receiver] = grad_rij

            fij = -2.0*dense.to("cpu").detach().numpy()
            fij[:,:,0] = 0.5*(fij[:,:,0] - fij[:,:,0].T)
            fij[:,:,1] = 0.5*(fij[:,:,1] - fij[:,:,1].T)
            fij[:,:,2] = 0.5*(fij[:,:,2] - fij[:,:,2].T)

        assert np.sum(np.abs(np.sum(fij,0) - atoms.calc.results["forces"])) < tols

        fijvi = np.zeros([fij.shape[0], fij.shape[1]])
        fijvi[:,:] = ( fij[:,:,0]*v_next[:,0] 
                      +fij[:,:,1]*v_next[:,1] 
                      +fij[:,:,2]*v_next[:,2])

    #A much slower version of force used for energy conservation
    # but gives the correct energy CV equations and so heat flux
    elif fijtype == "dUidrj":
        model = atoms.calc.models[0]
        batch_base = atoms.calc._atoms_to_batch(atoms)
        batch = atoms.calc._clone_batch(batch_base)
        out = model(batch.to_dict(), compute_stress=True, training=True)

        #Get dUi/drj as shown in Marcel et al
        positions = batch['positions']
        node_energy = out["node_energy"]
        dUidrj = torch.zeros((N, N, 3), device=positions.device, dtype=positions.dtype)
        for i in range(N-1):
            dUidrj[i,:,:] = torch.autograd.grad(node_energy[i], positions, 
                                                retain_graph=True, only_inputs=True)[0]
        #Final call with no retain graph to free memory
        dUidrj[N-1,:,:] = torch.autograd.grad(node_energy[N-1], positions, 
                                            retain_graph=False, only_inputs=True)[0]

        #Copy to CPU and delete GPU
        dUidrj = dUidrj.cpu().numpy()
        
        #This ensures Newton's 3rd law but is not 
        #consistent with energy conservation form       
        #fij = -(dUidrj - dUidrj.transpose(1, 0, 2))
        #assert np.sum(np.abs(np.sum(fij,0) - atoms.calc.results["forces"])) < 1e-8

        fij = -2.*dUidrj
        fijvi = np.zeros([fij.shape[0], fij.shape[1]])
        fijvi[:,:] = -2.*(  dUidrj[:,:,0]*v_next[:,0] 
                          + dUidrj[:,:,1]*v_next[:,1] 
                          + dUidrj[:,:,2]*v_next[:,2])

    elif fijtype == "dUidrj_opt":
        from torch.func import jacrev

        model = atoms.calc.models[0]
        batch_base = atoms.calc._atoms_to_batch(atoms)
        batch_dict = batch_base.to_dict()
        positions = batch_dict['positions'] # The input we want to differentiate wrt

        # 2. Define a functional wrapper
        def get_node_energies(pos):
            # Create a shallow copy of the batch dict to avoid side-effects
            # and inject the 'pos' tensor (which will be a Tracer during jacrev)
            batch_input = batch_dict.copy()
            batch_input['positions'] = pos
            
            # Run the model
            # We re-run the forward pass here so jacrev can trace it.
            # Note: training=True handles things like Dropout/BatchNorm if present.
            out = model(batch_input, compute_stress=False, compute_force=False, training=True)
            
            # Ensure output is 1D (N,) so jacrev produces a (N, N, 3) tensor.
            # If out["node_energy"] is (N, 1), the result would be (N, 1, N, 3).
            return out["node_energy"].squeeze()

        # 3. Compute the Jacobian using jacrev
        # This replaces the entire loop.
        # Output shape: (N_output, N_input, 3) -> (N, N, 3)
        dUidrj = jacrev(get_node_energies)(positions)

        #Copy to CPU and delete GPU
        dUidrj = dUidrj.cpu().numpy()
        
        #This ensures Newton's 3rd law but is not 
        #consistent with energy conservation form       
        #fij = -(dUidrj - dUidrj.transpose(1, 0, 2))
        #assert np.sum(np.abs(np.sum(fij,0) - atoms.calc.results["forces"])) < 1e-8

        fij = -2.*dUidrj
        fijvi = np.zeros([fij.shape[0], fij.shape[1]])
        fijvi[:,:] = -2.*(  dUidrj[:,:,0]*v_next[:,0] 
                          + dUidrj[:,:,1]*v_next[:,1] 
                          + dUidrj[:,:,2]*v_next[:,2])

    else:
        raise IOError("maceversion should be Custom or your installed system version >0.3.13")

    if timing:
        t2 = time.time()
        print("Get fij time=", t2-t1)

    # Get total momentum change of particles between planes (should bin this but need to check vs. mvbins)
    mv_MOP_planes = bin_MD(r, mv, Nbins, Lz)
    mv_MOP_hist.append(mv_MOP_planes)

    energy_MOP_planes = bin_MD(r, E, Nbins, Lz)
    energy_MOP_hist.append(energy_MOP_planes)

    ##############################
    # MOP kinetic calculation 
    # P^k(t) = \sum_i \boldsymbol{v}_{i} (t) (sgn(z_p - z_i(t+dt)) - sgn(z_p - z_i(t)))
    ##############################

    Nplanes = Nbins+1
    MOPstress_k, MOPenergy_k = get_MOP_kinetic(r, r_prev, mv, E, Lz, Nplanes)

    MOPstress_k_hist.append(MOPstress_k)
    MOPenergy_k_hist.append(MOPenergy_k)

    if timing:
        t3 = time.time()
        print("Kinetic + bin time=", t3-t2)

    ##############################
    # MOP Configuritonal calculation 
    # P^c = \sum_i \sum_j \boldsymbol{f}_{ij} (sgn(z_p - zi) - sgn(z_p - zj))
    ##############################

    #Optimized numba
    r_z = r[:, 2].astype(np.float64)  # Extract z-coordinates

    MOPstress_c, MOPenergy_c = get_MOP_stress_power(r_z, fij, fijvi, Lz, Nplanes)

    MOPstress_c_hist.append(MOPstress_c)
    MOPenergy_c_hist.append(MOPenergy_c)

    if timing:
        t4 = time.time()
        print("Config time=", t4-t3)

    #Get IK1 stress
    #Configurational
    Pc = atoms.get_stresses(include_ideal_gas=False)
    Pcbins.append(bin_MD(r, Pc, Nbins, Lz))
    #Kinetic
    Pk = np.zeros((len(atoms), 6))  # Voigt notation
    stresscomp = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])
    invmass = 1.0 / m
    for alpha in range(3):
        for beta in range(alpha, 3):
            Pk[:, stresscomp[alpha, beta]] -= (
                mv[:, alpha] * mv[:, beta] * invmass)
    Pkbins.append(bin_MD(r, Pk, Nbins, Lz))

    ##############################
    #Add some checks here
    ##############################
    if checks:
        #Empty bins must have balanced force on top and bottom 
        Nbin = bin_MD(r, np.ones(r.shape[0]), Nbins, Lz)
        Fds = MOPstress_c[1:,:]-MOPstress_c[:-1,:]
        binnos = np.where(Nbin == 0)
        for binno in binnos[0]:
            if  np.sum(np.abs(Fds[binno])) > 1e-6:
                print("Forces in empty bins", binno, Fds[binno])

        #Forces when a single molecule is in bin equal to forces over planes either side
        Nplanes = Nbins+1
        dz = Lz / Nbins
        z_planes = np.arange(Nplanes)*dz 
        binnos = np.where(Nbin == 1)
        F = atoms.calc.results["forces"]
        for binno in binnos[0]:
            indx = (z_planes[binno] < r[:,2]) & (r[:,2] < z_planes[binno+1])
            if  np.abs(np.sum(F[indx,:][0]-Fds[binno])) > 1e-5:
                print(binno, Nbin[binno], z_planes[binno], r[indx,2][0], z_planes[binno+1],
                      F[indx,:][0], Fds[binno], np.abs(np.sum(F[indx,:][0]-Fds[binno])))
        if timing:
            t5 = time.time()
            print("Checks time=", t5-t4)

    #Print energy to commandline and write xyz file
    printenergy(atoms, t)

    #Clean up extra allocated memory
    for name in ["dUidrj", "fij", "out", "node_energy", "batch", "batch_base", "positions", "fijvi"]:
        try:
            del globals()[name]
        except KeyError:
            pass
    torch.cuda.synchronize()


#Convert to arrays
Pi_c = np.array(MOPstress_c_hist)
Pi_k = np.array(MOPstress_k_hist)
mv_bin = np.array(mv_MOP_hist)

E_c = np.array(MOPenergy_c_hist).squeeze()
E_k = np.array(MOPenergy_k_hist).squeeze()
E_bin = np.array(energy_MOP_hist)
dEdt = np.diff(E_bin, axis=0)/dt

Pi_IK1_c = np.array(Pcbins)
Pi_IK1_k = np.array(Pkbins)

#Save data if slow to get
#if fijtype == "dUidrj":
#    import pickle
#    pickle.dump([MOPstress_c_hist, MOPstress_k_hist, MOPenergy_c_hist, MOPenergy_k_hist, energy_MOP_hist, Pcbins, Pkbins], open("duidrj.p", "bw+"))

c = 2

#Only worth plotting spatial values on long runs
if nsteps > 1000:
    #Plot Pzz as function of z - Fig 2
    plt.plot(np.mean(Pi_c[:,:,c],0), label="$\Pi^c_{_{MOP}}$")
    plt.plot(-np.mean(Pi_k[:,:,c],0)/dt, label="$\Pi^k_{_{MOP}}$")
    plt.plot(np.mean(Pi_c[:,:,c],0)-np.mean(Pi_k[:,:,c],0)/dt, label="$\Pi_{_{MOP}}$")

    plt.plot(np.mean(Pi_IK1_c[:,:,c],0), '--', label="$\Pi^c_{_{IK1}}$")
    plt.plot(-np.mean(Pi_IK1_k[:,:,c],0), '--', label="$\Pi^k_{_{IK1}}$")
    plt.plot(np.mean(Pi_IK1_c[:,:,c],0)-np.mean(Pi_IK1_k[:,:,c],0), '--', label="$\Pi_{_{IK1}}$")
    plt.legend()
    plt.show()

#Plot CV time evolution if results taken every timestep - Fig 3
if Nevery == 1:
    binno = 150
    ixyz = 0
    fig, axs = plt.subplots(2,1)
    Fds_c = Pi_c[:,binno+1,ixyz]-Pi_c[:,binno,ixyz]
    Fds_k = Pi_k[:,binno+1,ixyz]-Pi_k[:,binno,ixyz]
    dmvdt = np.diff(np.array(mv_MOP_hist)[:,binno,ixyz])/dt

    #Plot CV time evolution
    axs[0].plot(Fds_c[:-1], '--', zorder=4, label="$\Pi^c$"); 
    axs[0].plot(Fds_k[1:]/dt, label="$\Pi^k$"); 
    axs[0].plot(dmvdt[:], label=r"$\frac{d}{dt} \rho u $"); 
    axs[0].plot(Fds_c[:-1]-dmvdt[:]-Fds_k[1:]/dt, "k", lw=0.5, label=r"Sum"); 
    plt.legend()

    #Plot CV energy time evolution
    Eds_c = E_c[:,binno+1]-E_c[:,binno]
    Eds_k = E_k[:,binno+1]-E_k[:,binno]
    dedt = np.diff(np.array(energy_MOP_hist)[:,binno])/dt

    axs[1].plot(Eds_c[:-1] , '--', zorder=4, label="$f_{ij} v_i$"); 
    axs[1].plot(Eds_k[1:]/dt, label="$e_i v_i$"); 
    axs[1].plot(dedt[:], label=r"$\frac{d}{dt} \rho e_i $"); 
    axs[1].plot(Eds_c[:-1] -dedt[:]-Eds_k[1:]/dt, "k", lw=0.5, label=r"Sum"); 
    plt.legend()
    axs[1].set_ylim([-1.5,1.5])
    plt.show()

    #Plot just forces vs. d/dt with kinetic parts removed
    fig, axs = plt.subplots(2,1)

    axs[0].plot(Fds_c[:-2], '--', zorder=4, label="$\Pi^c$"); 
    axs[0].plot(dmvdt[:-1]+Fds_k[1:-1]/dt, label=r"$\frac{d}{dt} \rho u -\Pi^k$"); 
    axs[0].plot(Fds_c[:-2]-dmvdt[:-1]-Fds_k[1:-1]/dt, "k", lw=0.5, label=r"Sum"); 
    plt.legend()

    axs[1].plot(Eds_c[:-1], '--', zorder=4, label="$f_{ij} v_i$"); 
    axs[1].plot(dedt[:]+Eds_k[1:]/dt, label=r"$\frac{d}{dt} \rho e_i - e_i v_i$"); 
    axs[1].plot(Eds_c[:-1] -dedt[:]-Eds_k[1:]/dt, "k", lw=0.5, label=r"Sum"); 
    plt.legend()
    plt.show()

    #Note if you use verlet time integrator, values stored on timestep
    #so you need to shift plots by halfstep
    #tm = np.linspace(0,(nsteps-1)*dt,Fds_c.shape[0])
    #plt.plot(tm, Fds_c)
    #plt.plot(tm[:-1]+dt/2., dmvdt)


    ############################
    # Try reconstruction method
    ############################

    #Momentum reconstruction
    Pi_c_recon, mom_consv = reconstruct_and_plot(mv_bin[:,:,ixyz], -Pi_k[:,:,ixyz], dt, plottype="Momentum", binno=binno, ref=Pi_c[:,:,ixyz])

    #Conservation errors near wall throw agreement, try just in center
    Pi_c_recon = reconstruct_energy_flux_ref(mv_bin[:,100:300,ixyz], -Pi_k[:,100:301,ixyz], dt, reference=Pi_c[:,100,ixyz], periodic=False)
    plt.plot(Pi_c_recon[49,:], 'r-')
    plt.plot(Pi_c[49,100:301,0], 'b--')
    plt.show()

    plt.plot(Pi_c_recon[:,100], 'r-')
    plt.plot(Pi_c[:,200,0], 'b--')
    plt.show()

    #Energy reconstruction
    E_c_recon, E_consv = reconstruct_and_plot(E_bin, -E_k, dt, binno=binno, ref=E_c)

    Eds_c = E_c[:,1:]-E_c[:,:-1]
    Eds_k = E_k[:,1:]-E_k[:,:-1]
    bI = dEdt[:,:]+Eds_k[1:,:]/dt

    E_c_recon = reconstruct_energy_flux_ref(E_bin[:,100:300], -E_k[:,100:301], dt, reference=E_c[:,100], periodic=False)
    plt.plot(E_c_recon[49,:], 'r-')
    plt.plot(E_c[49,100:301], 'b--')
    plt.show()

    plt.plot(E_c_recon[:,100], 'r-')
    plt.plot(E_c[:,200], 'b--')
    plt.show()

