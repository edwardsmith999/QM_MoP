import numpy as np
import time
import torch
import sys
import os

#sys.path.insert(1, '/home/es205/codes/ase')
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen, Inhomogeneous_NPTBerendsen
from ase.io import read, write, vasp
from ase.optimize import BFGS

from MOP import get_MOP_kinetic, get_MOP_stress_power, bin_MD
from utils import printenergy_and_pressure, check_bidirectional_graph, get_atom_potential_energies

from_voigt = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])
to_voigt = [[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
maceversion = "system"
if maceversion == "system":

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
elif maceversion == "custom":
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
    tols = 1e-8 #Runs at 64 bit so more accurate
else:
    pass

from mace.calculators import MACECalculator

AtomDict = {"O":8, "H":1}

Nbins = 30
savefreq = 10

outdir = "./results/"
intialfiles = "./initial_states/"

read_vasp = False
minimise = False
dynamics = "NPT"
extended_stats = True
Tset = 600 #K
Pset = 10000.0 #bar #1.01325 #bar
dt = 0.5
tols = 1e-2
Ntplot = 10

#Start from VASP file to start or initial state
#which is equilibrated with velocities applied
if read_vasp:
    atoms = vasp.read_vasp(intialfiles + "water.vasp")
    MaxwellBoltzmannDistribution(atoms, temperature_K=Tset)
    #Remove drift velocity
    Stationary(atoms)
else:
    atoms = read(intialfiles + "water.traj")
atoms.wrap()

# Optionally make system (8 times) bigger
#atoms = atoms.repeat((2, 2, 2))

#Get system sizes
N = len(atoms)

#Define mace calculator
modelpath = './foundations_models/'
atoms.calc = MACECalculator(modelpath+"mace-mpa-0-medium.model", 
                            device='cuda', default_dtype="float32",
                            enable_cueq=True, #Previously worked with cuequivariance-torch==0.3.0 and e3nn==0.4.4
                            compile_mode="default")   

#Minimize liquid atoms
if minimise:
    #Minimise whole system
    dyn = BFGS(atoms, trajectory='H2O.traj')
    dyn.run(fmax=0.3)
    atoms.write("opt_water.traj")

if dynamics == "NVT":
    dyn = NoseHooverChainNVT(atoms, dt*units.fs, Tset, 100*dt*units.fs) 
elif dynamics == "NPT":
    print(dt*units.fs, Tset, Pset*units.bar)
    #dyn = NPT(atoms, dt*units.fs, temperature_K=Tset, 
    #          externalstress=Pset*units.bar)
    #dyn = NPTBerendsen(atoms, timestep=0.1 * units.fs, temperature_K=Tset,
    #                   taut=100 * units.fs, pressure_au=Pset * units.bar,
    #                   taup=1000 * units.fs, compressibility_au=4.57e-5 / units.bar)
    dyn = Inhomogeneous_NPTBerendsen(atoms, timestep=dt * units.fs, temperature_K=Tset, mask=(1,1,0),
                                     taut=100 * units.fs, pressure_au=Pset * units.bar,
                                     taup=1000 * units.fs, compressibility_au=4.57e-5 / units.bar)

else:
    dyn = VelocityVerlet(atoms, dt*units.fs)


#Check if graph is bidirectional
#batch_base = atoms.calc._atoms_to_batch(atoms)
#batch = atoms.calc._clone_batch(batch_base)
#sender, receiver = batch["edge_index"]
#is_bidir, missing, fwd, rev = check_bidirectional_graph(batch["edge_index"])

# Now run the dynamics
dyn.atoms.write(outdir+'mace_run.xyz', append=False)
countbins = {k: [] for k in AtomDict}
mbins = {k: [] for k in AtomDict}
KEbins = {k: [] for k in AtomDict}
mvbins = {k: [] for k in AtomDict}
Pbins = {k: [] for k in AtomDict}
PEbins = {k: [] for k in AtomDict}

MOPstress_c_hist = []
MOPstress_k_hist = []
mv_MOP_hist = []

Pcbins = []
Pkbins = []
fijvi = np.zeros([N,N])

r = atoms.get_positions()

logfile = outdir + "printenergy.log"
if os.path.isfile(logfile):
    os.remove(logfile)
for t in range(20000):

    #Print energy to commandline and write xyz file
    printenergy_and_pressure(t, atoms, logfile)

    #Loop Ntplot then do one timestep for Pk calculation
    dyn.run(Ntplot-1)
    r_prev = atoms.get_positions()
    dyn.run(1)

    #For an NPT, these need to be recalculated
    Lz = atoms.cell[2][2]
    V =  atoms.get_volume()
    binrange = np.linspace(0, Lz, Nbins)

    #Get atom properties
    atoms.wrap()
    atomtype = atoms.get_chemical_symbols()
    atomno = atoms.get_atomic_numbers()
    m = atoms.get_masses()
    r = atoms.get_positions()
    mv = atoms.get_momenta()
    v = np.array([mv[:,i]/m for i in range(3)]).T

    #Kinetic Energy/Temperature
    KE = 0.5*m*np.sum(v**2, axis=1)
    PE = get_atom_potential_energies(atoms)
    E = KE + PE

    #Remove drift velocity
    #Stationary(atoms)
    #print("step=", t, "Drift v=", np.sum(mv,0), np.sum(atoms.get_momenta(),0))

    try:
        #Kinetic Energy/Temperature
        v = np.array([mv[:,i]/m for i in range(3)]).T
        KE = m*np.sum(v**2, axis=1)
        #Check sum of local temperature adds to total
        assert abs(atoms.get_temperature() - np.sum(KE) / (3 * N * units.kB)) < 1e-5

        #A low level call to get force per atom
        model = atoms.calc.models[0]
        batch_base = atoms.calc._atoms_to_batch(atoms)
        batch = atoms.calc._clone_batch(batch_base)

        if maceversion == "custom":
            fij = 2.0*atoms.calc.results["fij"]
            fij[:,:,0] = 0.5*(fij[:,:,0] - fij[:,:,0].T)
            fij[:,:,1] = 0.5*(fij[:,:,1] - fij[:,:,1].T)
            fij[:,:,2] = 0.5*(fij[:,:,2] - fij[:,:,2].T)

            Pc = atoms.get_stresses(include_ideal_gas=False)

        #This version work with  MACE 0.3.14 as it gets fij from edge_forces
        #however it uses hidden functions like "calc._atoms_to_batch"
        #which might change in updates to MACE. 
        elif maceversion == "system":

            #A low level call to get force per atom
            model = atoms.calc.models[0]
            batch_base = atoms.calc._atoms_to_batch(atoms)
            batch = atoms.calc._clone_batch(batch_base)
            out = model(batch.to_dict(), compute_stress=True, compute_atomic_stresses=True,
                        compute_edge_forces=True, training=True)
            grad_rij = out["edge_forces"]
            dense = torch.zeros((N, N, grad_rij.shape[1]), device=grad_rij.device, dtype=grad_rij.dtype)
            sender, receiver = batch["edge_index"]
            dense[sender, receiver] = grad_rij

            fij = -2.0*dense.to("cpu").detach().numpy()
            fij[:,:,0] = 0.5*(fij[:,:,0] - fij[:,:,0].T)
            fij[:,:,1] = 0.5*(fij[:,:,1] - fij[:,:,1].T)
            fij[:,:,2] = 0.5*(fij[:,:,2] - fij[:,:,2].T)
        
            #Calculate virial by hand to check
            dense_rij = torch.zeros((N, N, grad_rij.shape[1]), device=grad_rij.device, dtype=grad_rij.dtype)
            dense_rij[sender, receiver] = batch["positions"][receiver,:] - batch["positions"][sender,:] + batch["shifts"]
            rij = dense_rij.to("cpu").detach().numpy()
            virial = -0.5 * np.einsum('ijk,ijl->kl', rij, fij)/V
            assert np.sum(np.abs(virial - atoms.get_stress(include_ideal_gas=False)[from_voigt])) < 1e-5

            Pc = -out["atomic_virials"].cpu().detach().numpy()[:,to_voigt[0],to_voigt[1]]

        #Check both forces add up and configurational stresses
        assert np.sum(np.abs(np.sum(fij,0) - atoms.calc.results["forces"])) < tols
        assert np.sum(np.abs(np.sum(Pc,0)/V - atoms.get_stress(include_ideal_gas=False))) < 1e-4

        #Save in bins
        for k, v in AtomDict.items():
            mask = atomno == v

            #Bin MD data
            countbins[k].append(bin_MD(r, np.ones(r.shape[0]), Nbins, Lz, mask))
            mbins[k].append(bin_MD(r, m, Nbins, Lz, mask))
            KEbins[k].append(bin_MD(r, KE, Nbins, Lz, mask))
            mvbins[k].append(bin_MD(r, mv, Nbins, Lz, mask))


        # Get total momentum change of particles between planes (should bin this but need to check vs. mvbins)
        mv_MOP_planes = bin_MD(r, mv, Nbins, Lz)
        mv_MOP_hist.append(mv_MOP_planes)

        ##############################
        # MOP kinetic calculation 
        # P^k(t) = \sum_i m_i \boldsymbol{v}_{i} (t) (sgn(z_p - z_i(t+dt)) - sgn(z_p - z_i(t)))
        ##############################

        Nplanes = Nbins+1
        MOPstress_k, MOPenergy_k = get_MOP_kinetic(r, r_prev, mv, E, Lz, Nplanes)
        MOPstress_k_hist.append(MOPstress_k)

        ##############################
        # MOP Configuritonal calculation 
        # P^c = \sum_i \sum_j \boldsymbol{f}_{ij} (sgn(z_p - zi) - sgn(z_p - zj))
        ##############################
        r_z = r[:, 2].astype(np.float64)  # Extract z-coordinates
        MOPstress_c, MOPenergy_c = get_MOP_stress_power(r_z, fij, fijvi, Lz, Nbins)
        MOPstress_c_hist.append(MOPstress_c)

        #Get IK1 stress
        #Configurational
        Pcbin = bin_MD(r, Pc, Nbins, Lz)
        assert np.sum(np.abs(np.sum(Pc, axis=0) - np.sum(Pcbin, axis=0))) < 1e-5
        Pcbins.append(Pcbin)

        #Kinetic
        Pk = np.zeros((len(atoms), 6))  # Voigt notation
        invmass = 1.0 / m
        for alpha in range(3):
            for beta in range(alpha, 3):
                Pk[:, from_voigt[alpha, beta]] -= (
                    mv[:, alpha] * mv[:, beta] * invmass)

        #Check kinetic pressure vs. manual calc
        assert np.sum(np.abs(np.sum(Pk,0) - atoms.get_kinetic_stress()*V)) < 1e-8

        #Check bins add up
        Pkbin = bin_MD(r, Pk, Nbins, Lz)
        assert np.sum(np.abs(np.sum(Pk, axis=0) - np.sum(Pkbin, axis=0))) < 1e-8
        Pkbins.append(Pkbin)

        #Final check on pressure with print
        assert np.sum(np.abs(np.sum(Pk+Pc,0)/V - atoms.get_stress(include_ideal_gas=True))) < 1e-4

    except AssertionError as e:
        print("Assertion error", e)

    if t % savefreq == 0:
        for k in AtomDict:
            np.save(outdir+f"countbins.npy", np.array(countbins))
            np.save(outdir+f"mbins.npy", np.array(mbins))
            np.save(outdir+f"KEbins.npy", np.array(KEbins))
            np.save(outdir+f"mvbins.npy", np.array(mvbins))
            if extended_stats:
                np.save(outdir+f"Pbins.npy", np.array(Pbins))
                np.save(outdir+f"PEbins.npy", np.array(PEbins))
        write(outdir+"water_checkpoint{:05}.traj".format(t), atoms)

        np.save(outdir+f"MOPstress_c_hist.npy", np.array(MOPstress_c_hist))
        np.save(outdir+f"MOPstress_k_hist.npy", np.array(MOPstress_k_hist))
        np.save(outdir+f"Pcbins.npy", np.array(Pcbins))
        np.save(outdir+f"Pkbins.npy", np.array(Pkbins))
        np.save(outdir+f"mv_MOP_hist.npy", np.array(mv_MOP_hist))
        dyn.atoms.write(outdir+'mace_run.xyz', append=True)

