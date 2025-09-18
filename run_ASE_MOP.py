import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import sys

from scipy.stats import binned_statistic
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit

from ase import units
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.io import read, write, vasp
from ase.calculators.calculator import PropertyNotImplementedError

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    print("Failed to import Numba, MoP routines should still run but much slower")
    NUMBA_AVAILABLE = False

def optional_njit(*njit_args, **njit_kwargs):
    if NUMBA_AVAILABLE:
        return njit(*njit_args, **njit_kwargs)
    else:
        # fallback: just return function unchanged
        def wrapper(func):
            return func
        return wrapper


def printenergy(a, t):
    """Function to print the potential, kinetic and total energy"""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    f = a.get_forces()
    m = a.get_masses()
    mv = a.get_momenta()
    v = np.array([mv[:,i]/m for i in range(3)]).T
    print(t, 'Energy per atom: Epot = %.7f eV  Ekin = %.7f eV (T=%3.0fK)  '
          'Etot = %.7f eV sum Fi vi = %.7f eV' % (epot, ekin, ekin / (1.5 * units.kB), 
           epot + ekin, np.sum(f[:,0]*v[:,0]+f[:,1]*v[:,1]+f[:,2]*v[:,2])))


#@njit(fastmath=True, cache=True)
@optional_njit(fastmath=True, cache=True)
def get_MOP_stress_power(r_z, fij, fijvi, Lz, Nbins, threshold=1e-7):
    """
    Simple Numba version 
    Returns MOPstress_c array
    """
    n_atoms = r_z.shape[0]
    n_dims = fij.shape[2]
    MOPstress_c = np.zeros((Nbins+1, n_dims))  # +1 for safety
    #Add power for energy calculation
    MOPpower_c = np.zeros((Nbins+1,1))  # +1 for safety
    dz = Lz / Nbins
    
    for i in range(n_atoms):
        for j in range(n_atoms):
            # Check threshold first
            force_magnitude = fij[i, j, 0]
            if force_magnitude > threshold or force_magnitude < -threshold:
                z1, z2 = r_z[i], r_z[j]
                
                # Compute bin indices using floor division
                i1 = np.int32(z1 / dz)  # Use division instead of floor division
                i2 = np.int32(z2 / dz)
                
                # Handle periodic boundary conditions
                direct_delta = (i2 - i1) % Nbins
                wrap_delta = (i1 - i2) % Nbins
                
                if direct_delta <= wrap_delta:
                    direction = -1
                    # Add contributions to crossed bins
                    for k in range(direct_delta):
                        bin_idx = (i1 + 1 + k) % Nbins
                        for dim in range(n_dims):
                            MOPstress_c[bin_idx, dim] += 0.5 * fij[i, j, dim] * direction
 
                        MOPpower_c[bin_idx] += 0.5 * fijvi[i,j] * direction
                else:
                    direction = 1
                    for k in range(wrap_delta):
                        bin_idx = (i1 - k) % Nbins
                        for dim in range(n_dims):
                            MOPstress_c[bin_idx, dim] += 0.5 * fij[i, j, dim] * direction

                        MOPpower_c[bin_idx] += 0.5 * fijvi[i,j] * direction

    return MOPstress_c, MOPpower_c

def bin_MD(r, A, Nbins=10, Lz=1., mask=None):
    """
    Bin a given input array A (e.g., mass, kinetic energy, momentum, etc.) based on the z-coordinates.
    
    Parameters:
    r      : (N, 3) array of atomic positions.
    A      : Input array (e.g., mass, KE, momentum). The shape determines how to bin it.
    Nbins  : Number of bins (default: 10).
    Lz     : Length of the system in the z-direction (default: 1.0).
    mask   : Boolean array selecting the atoms to include. If None, includes all atoms.
    
    Returns:
    Binned array corresponding to A.
    """
    if mask is None:
        mask = np.ones(r.shape[0], dtype=bool)
    
    if A.ndim == 1:
        # Scalar quantity (e.g., mass, KE, PE)
        return binned_statistic(r[mask, 2], A[mask], statistic='sum', bins=Nbins, range=[0, Lz]).statistic
    elif A.ndim == 2:
        # Vector quantity (e.g., momentum, pressure tensor)
        return np.array([
            binned_statistic(r[mask, 2], A[mask, j], statistic='sum', bins=Nbins, range=[0, Lz]).statistic
            for j in range(A.shape[1])
        ]).T
    else:
        raise ValueError("Unsupported array shape for binning.")


def get_force_lowlevel(atoms, pairwise=True):

    """
        A low level call to get force per atom
        using an explict autograd operation
    """

    if maceversion != "custom":
        raise ImportError("Version of mace must be custom from edwardsmith999 to use this low level interface")

    model = atoms.calc.models[0]
    batch_base = atoms.calc._atoms_to_batch(atoms)
    batch = atoms.calc._clone_batch(batch_base)
    out = model(batch.to_dict(), compute_stress=True, training=True)
    total_energy = out['energy'].sum()  # or sum of node energies
    if pairwise:
        rij = out["vectors"]
        grad_rij = torch.autograd.grad(total_energy, rij, retain_graph=True)[0]
        dense = torch.zeros((N, N, grad_rij.shape[1]), device=grad_rij.device, dtype=grad_rij.dtype)
        sender, receiver = batch["edge_index"]
        dense[sender, receiver] = grad_rij

        fij = -2.0*dense.to("cpu").numpy()
        fij[:,:,0] = 0.5*(fij[:,:,0] - fij[:,:,0].T)
        fij[:,:,1] = 0.5*(fij[:,:,1] - fij[:,:,1].T)
        fij[:,:,2] = 0.5*(fij[:,:,2] - fij[:,:,2].T)

        if checks:
            assert np.sum(np.abs(np.sum(fij,0) - atoms.calc.results["forces"])) < 1e-8
        return fij
    else:
        positions = batch['positions'].requires_grad_(True)
        forces = -torch.autograd.grad(total_energy, positions, retain_graph=True)[0]

        return forces


def get_atom_potential_energies(atoms):

    try:
        #Only implemented currently in custom version
        PE = atoms.get_potential_energies()
    except PropertyNotImplementedError:
        #Otherwise get e0 energy and subtract from node energy
        batch_base = atoms.calc._atoms_to_batch(atoms)
        batch = atoms.calc._clone_batch(batch_base)
        node_heads = batch["head"][batch["batch"]]
        num_atoms_arange = torch.arange(batch["positions"].shape[0])
        node_e0 = atoms.calc.models[0].atomic_energies_fn(batch["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        PE = atoms.calc.results["node_energy"]+node_e0.cpu().numpy()

    return PE


#Define all simulation parameters here
AtomDict = {"Zr":40, "O":8, "H":1}

nsteps = 20
Nbins = 400
savefreq = 1
Nevery = 1

timing = False
checks = False
read_vasp = False

outdir = "./results/"
modelpath = "./foundation_models/"

dynamics = "leapfrog" # "verlet" or "leapfrog"
maceversion = "custom" #system or "custom"
fijtype = "dUdrij"   # "dUidrj" or "dUdrij"

Tset = 500 #Temperature
dt = 0.5

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
    atoms = vasp.read_vasp("water_ZrO2_initial_doubled.vasp")
    MaxwellBoltzmannDistribution(atoms, temperature_K=Tset)
    #Remove drift velocity
    Stationary(atoms)
else:
    atoms = read("500K.traj")

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
    dyn = VelocityVerlet(atoms, dt*units.fs)
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
        mv += dt*units.fs * f

        atoms.set_momenta(mv)
        r_prev = r.copy() # Save previous positions before updating
        r[:,0] += dt*units.fs * mv[:,0]/m[:]
        r[:,1] += dt*units.fs * mv[:,1]/m[:]
        r[:,2] += dt*units.fs * mv[:,2]/m[:]
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
    mv_next = mv + 0.5*dt*units.fs * f
    v_next = np.array([mv_next[:,i]/m for i in range(3)]).T

    #Remove drift velocity
    #Stationary(atoms)
    #print("step=", t, "Drift v=", np.sum(mv,0), np.sum(atoms.get_momenta(),0))

    #Kinetic Energy/Temperature
    KE = 0.5*m*np.sum(v**2, axis=1)
    PE = get_atom_potential_energies(atoms)
    E = KE + PE
    dE_dt[t] = np.sum((E - E_prev) / (dt * units.fs))

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

    else:
        raise IOError("maceversion should be Custom or >0.3.13")


    ########################################################

    ########################################################

    if timing:
        t2 = time.time()
        print("Get fij time=", t2-t1)

    #Define a range of Nplanes evenly spaced planes in z direction (r[:,2]) 
    #filling the domain and create 
    #an array of size MOPstress(Nplanes, 3) for them
    Nplanes = Nbins+1
    dz = Lz / Nbins
    z_planes = np.arange(Nplanes)*dz 

    # Get total momentum change of particles between planes (should bin this but need to check vs. mvbins)
    mv_MOP_planes = bin_MD(r, mv, Nbins, Lz)
    mv_MOP_hist.append(mv_MOP_planes)

    energy_MOP_planes = bin_MD(r, E, Nbins, Lz)
    energy_MOP_hist.append(energy_MOP_planes)

    ##############################
    # MOP kinetic calculation 
    # P^k(t) = \sum_i \boldsymbol{v}_{i} (t) (sgn(z_p - z_i(t+dt)) - sgn(z_p - z_i(t)))
    ##############################
    MOPstress_k = np.zeros((Nplanes, 3))
    MOPenergy_k = np.zeros((Nplanes, 1))
    # Determine plane crossings and calculate momentum contributions
    for i in range(len(r)):
        # Get min and max plane so can only check
        # planes between the old and new positions
        z_bin = np.digitize(r[i,2], bins=z_planes)-1
        z_prev_bin = np.digitize(r_prev[i, 2], bins=z_planes)-1

        # Check for crossings with each plane
        if z_bin == z_prev_bin:
            continue
        else:
            for b in (min(z_bin, z_prev_bin), max(z_bin, z_prev_bin)):
                # If sign changes (crossing occurred), add momentum contribution
                cross = (  np.sign(z_planes[b] - r_prev[i, 2]) 
                         - np.sign(z_planes[b] - r[i, 2])) 
                MOPstress_k[b] += mv[i] * cross
                MOPenergy_k[b] += E[i]  * cross 

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

    MOPstress_c, MOPenergy_c = get_MOP_stress_power(r_z, fij, fijvi, Lz, Nbins)

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

E_c = np.array(MOPenergy_c_hist)
E_k = np.array(MOPenergy_k_hist)

Pi_IK1_c = np.array(Pcbins)
Pi_IK1_k = np.array(Pkbins)

c = 2

#Only worth plotting spatial values on long runs
if nsteps > 1000:
    #Plot Pzz as function of z - Fig 2
    plt.plot(np.mean(Pi_c[:,:,c],0), label="$\Pi^c_{_{MOP}}$")
    plt.plot(-np.mean(Pi_k[:,:,c],0)/units.fs, label="$\Pi^k_{_{MOP}}$")
    plt.plot(np.mean(Pi_c[:,:,c],0)-np.mean(Pi_k[:,:,c],0)/units.fs, label="$\Pi_{_{MOP}}$")

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
    dmvdt = np.diff(np.array(mv_MOP_hist)[:,binno,ixyz])/(dt*units.fs)

    #Plot CV time evolution
    axs[0].plot(Fds_c[:-1], '--', zorder=4, label="$\Pi^c$"); 
    axs[0].plot(Fds_k[1:]/units.fs, label="$\Pi^k$"); 
    axs[0].plot(dmvdt[:], label=r"$\frac{d}{dt} \rho u $"); 
    axs[0].plot(Fds_c[:-1]-dmvdt[:]-Fds_k[1:]/units.fs, "k", lw=0.5, label=r"Sum"); 
    plt.legend()

    #Plot CV energy time evolution
    Eds_c = E_c[:,binno+1,0]-E_c[:,binno,0]
    Eds_k = E_k[:,binno+1,0]-E_k[:,binno,0]
    dedt = np.diff(np.array(energy_MOP_hist)[:,binno])/(dt*units.fs)

    axs[1].plot(0.5*(Eds_c[:-1] + Eds_c[1:]), '--', zorder=4, label="$f_{ij} v_i$"); 
    axs[1].plot(Eds_k[1:]/units.fs, label="$e_i v_i$"); 
    axs[1].plot(dedt[:], label=r"$\frac{d}{dt} \rho e_i $"); 
    axs[1].plot(0.5*(Eds_c[:-1] + Eds_c[1:])-dedt[:]-Eds_k[1:]/units.fs, "k", lw=0.5, label=r"Sum"); 
    plt.legend()
    axs[1].set_ylim([-1.5,1.5])
    plt.show()

    #Plot just forces vs. d/dt with kinetic parts removed
    fig, axs = plt.subplots(2,1)

    axs[0].plot(Fds_c[:-2], '--', zorder=4, label="$\Pi^c$"); 
    axs[0].plot(dmvdt[:-1]+Fds_k[1:-1]/units.fs, label=r"$\frac{d}{dt} \rho u -\Pi^k$"); 
    axs[0].plot(Fds_c[:-2]-dmvdt[:-1]-Fds_k[1:-1]/units.fs, "k", lw=0.5, label=r"Sum"); 
    plt.legend()

    axs[1].plot(0.5*(Eds_c[1:]+Eds_c[:-1]), '--', zorder=4, label="$f_{ij} v_i$"); 
    axs[1].plot(dedt[:]+Eds_k[1:]/units.fs, label=r"$\frac{d}{dt} \rho e_i - e_i v_i$"); 
    axs[1].plot(0.5*(Eds_c[:-1] + Eds_c[1:])-dedt[:]-Eds_k[1:]/units.fs, "k", lw=0.5, label=r"Sum"); 
    plt.legend()
    plt.show()

    #Note if you use verlet time integrator, values stored on timestep
    #so you need to shift plots by halfstep
    #tm = np.linspace(0,(nsteps-1)*dt,Fds_c.shape[0])
    #plt.plot(tm, Fds_c)
    #plt.plot(tm[:-1]+dt/2., dmvdt)




