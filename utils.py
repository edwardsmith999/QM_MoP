import numpy as np
import torch
import sys
import os

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr

#sys.path.insert(1, '/home/es205/codes/ase')
from ase import units
from ase.calculators.calculator import PropertyNotImplementedError

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


def printenergy_and_pressure(i, a, logfile=None):
    """Function to print the potential, kinetic and total energy"""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    stress= a.get_stress(include_ideal_gas=True)
    Pk = a.get_kinetic_stress()
    Lx = a.cell[0][0]
    Ly = a.cell[1][1]
    Lz = a.cell[2][2]
    V =  a.get_volume()

    # Hydrostatic pressure in eV/Å³ (minus sign: see explanation below)
    P_eVA3 = - (stress[0] + stress[1] + stress[2]) / 3.0
    Pk_eVA3 = - (Pk[0] + Pk[1] + Pk[2]) / 3.0

    # Convert using ASE built-in conversion: 1 bar = units.bar (in eV/Å³)
    P_bar = P_eVA3 / units.bar
    Pk_bar = Pk_eVA3 / units.bar
    P_bar_components = -stress / units.bar
    labels = ['Pxx', 'Pyy', 'Pzz', 'Pyz', 'Pxz', 'Pxy']
    printstr = ('Step= %i Epot= %.3feV Ekin= %.3feV T= %3.0fK '
                'Etot= %.3feV Lx= %.3f Ly= %.3f Lz= %.3f V= %.3f P= %.3f Pk= %.3f Pc= %.3f ' % (
                i, epot, ekin, ekin / (1.5 * units.kB), epot + ekin, 
                Lx, Ly, Lz, V, P_bar, Pk_bar, P_bar-Pk_bar))
    printstr += f" ".join(f"{l}= {P_bar_components[i]:.3f}" for i, l in enumerate(labels))
    print(printstr)
    if logfile:
        with open(logfile, 'a') as f:
            f.write(printstr+"\n")

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


def check_bidirectional_graph(edge_index, N=None, verbose=True):
    """
    Check if a graph is bidirectional, i.e., for every edge (i,j) there exists (j,i).
    This means N(i) = N^{-1}(i) for all atoms i.
    
    Parameters:
    -----------
    edge_index : torch.Tensor or tuple
        Either a (2, num_edges) tensor or tuple of (sender, receiver)
    N : int, optional
        Number of atoms. If None, inferred from edge_index
    verbose : bool
        Whether to print detailed information
    
    Returns:
    --------
    is_bidirectional : bool
    missing_edges : list of tuples
        Edges that exist in one direction but not the reverse
    """
    
    # Handle different input formats
    if isinstance(edge_index, torch.Tensor):
        if edge_index.shape[0] == 2:
            sender = edge_index[0].cpu().numpy()
            receiver = edge_index[1].cpu().numpy()
        else:
            raise ValueError("edge_index should be (2, num_edges)")
    else:
        sender, receiver = edge_index
        if isinstance(sender, torch.Tensor):
            sender = sender.cpu().numpy()
            receiver = receiver.cpu().numpy()
    
    # Infer number of atoms if not provided
    if N is None:
        N = max(sender.max(), receiver.max()) + 1
    
    # Create edge set for fast lookup
    edges = set(zip(sender, receiver))
    num_edges = len(edges)
    
    # Check for reverse edges
    missing_edges = []
    for i, j in edges:
        if (j, i) not in edges:
            missing_edges.append((i, j))
    
    is_bidirectional = len(missing_edges) == 0
    
    # Build N(i) and N^{-1}(i) for each atom
    forward_neighbors = {i: set() for i in range(N)}  # N(i): i is sender
    reverse_neighbors = {i: set() for i in range(N)}  # N^{-1}(i): i is receiver
    
    for s, r in zip(sender, receiver):
        forward_neighbors[s].add(r)
        reverse_neighbors[r].add(s)
    
    # Check if N(i) == N^{-1}(i) for all atoms
    atoms_with_mismatch = []
    for i in range(N):
        if forward_neighbors[i] != reverse_neighbors[i]:
            atoms_with_mismatch.append(i)
    
    if verbose:
        print(f"Graph Statistics:")
        print(f"  Number of atoms: {N}")
        print(f"  Number of edges: {num_edges}")
        print(f"  Bidirectional: {is_bidirectional}")
        print(f"  Missing reverse edges: {len(missing_edges)}")
        
        if not is_bidirectional:
            print(f"\n  First few missing edges:")
            for edge in missing_edges[:5]:
                print(f"    Edge {edge} exists but reverse {edge[::-1]} does not")
        
        print(f"\n  Atoms where N(i) != N^(-1)(i): {len(atoms_with_mismatch)}")
        if atoms_with_mismatch and len(atoms_with_mismatch) <= 5:
            for i in atoms_with_mismatch:
                print(f"    Atom {i}:")
                print(f"      N(i) = {forward_neighbors[i]}")
                print(f"      N^(-1)(i) = {reverse_neighbors[i]}")
                print(f"      Difference: N(i) - N^(-1)(i) = {forward_neighbors[i] - reverse_neighbors[i]}")
                print(f"                  N^(-1)(i) - N(i) = {reverse_neighbors[i] - forward_neighbors[i]}")
    
    return is_bidirectional, missing_edges, forward_neighbors, reverse_neighbors

def verify_neighbor_sets(forward_neighbors, reverse_neighbors, verbose=True):
    """
    Verify that N(i) = N^{-1}(i) for all atoms.
    
    Parameters:
    -----------
    forward_neighbors : dict
        N(i) for each atom i (atoms where i is sender)
    reverse_neighbors : dict
        N^{-1}(i) for each atom i (atoms where i is receiver)
    verbose : bool
        Whether to print information
    
    Returns:
    --------
    all_equal : bool
    """
    all_equal = True
    mismatches = []
    
    for i in forward_neighbors.keys():
        if forward_neighbors[i] != reverse_neighbors[i]:
            all_equal = False
            mismatches.append(i)
    
    if verbose:
        if all_equal:
            print("✓ N(i) = N^(-1)(i) for all atoms")
        else:
            print(f"✗ N(i) != N^(-1)(i) for {len(mismatches)} atoms")
            if len(mismatches) <= 5:
                print(f"  Mismatched atoms: {mismatches}")
    
    return all_equal



def solve_timestep_ls(rhs, target_values, weights=False):
    """Solve least squares problem for a single timestep."""
    
    # Build system
    Nbins = rhs.shape[0]
    Nplanes =  target_values.shape[0]
    Nconstraints = Nbins + Nplanes
    A = lil_matrix((Nconstraints, Nplanes))
    b = np.zeros(Nconstraints)
    
    row = 0
    
    # Conservation: flux_c[i+1] - flux_c[i] = rhs[i]
    for i in range(Nbins):
        A[row, i] = -1.0
        A[row, i+1] = 1.0
        b[row] = rhs[i]
        row += 1
       
    # Known values
    if weights:
        weights = 1./Nbins
    else:
        weights = 1.

    for i in range(target_values.shape[0]):
        A[row, i] = 1.0 * weights
        b[row] = target_values[i]*weights
        row += 1
    
    # Solve
    A_csr = A.tocsr()
    result = lsqr(A_csr, b, atol=1e-10, btol=1e-10)
    
    return result[0], result[3]**2


def admal_tadmor_global_ls(positions, forces, edge_index, shifts):
    """
    Global least-squares Admal–Tadmor central-force decomposition.
    using   

    """

    N = positions.shape[0]
    M = edge_index.shape[1]

    # --- assemble sparse matrix ---
    A = lil_matrix((3 * N, M))
    F = forces.reshape(-1)

    for col in range(M):
        i,j = edge_index[:,col]
        rij = positions[j] - positions[i] + shifts[col]
        r = np.linalg.norm(rij)
        if r == 0.0:
            continue
        rhat = rij / r

        # atom i contribution
        for k in range(3):
            A[3*i + k, col] += rhat[k]

        # atom j contribution
        for k in range(3):
            A[3*j + k, col] -= rhat[k]

    A = A.tocsr()

    # --- solve least squares ---
    sol = lsqr(A, F)
    s = sol[0]

    # --- reconstruct central forces ---
    fij = np.zeros((N, N, 3))

    for col in range(M):
        i,j = edge_index[:,col]
        rij = positions[j] - positions[i] + shifts[col]
        rhat = rij / np.linalg.norm(rij)
        fvec = s[col] * rhat

        fij[i, j] = fvec
        fij[j, i] = -fvec

    return fij



def check_torque_conservation(atoms_in):

    # 1. Create a copy to avoid messing up your main simulation
    test_atoms = atoms_in.copy()
    
    # 2. CRITICAL: Turn off PBCs. 
    # This treats the atoms as a floating cluster in vacuum.
    test_atoms.set_pbc(False)
    
    # 3. Attach the calculator
    test_atoms.calc = atoms_in.calc
    
    # 4. Get forces
    forces = test_atoms.get_forces()
    positions = test_atoms.get_positions()
    
    # 5. Calculate Center of Mass (COM)
    # Measuring torque relative to COM is standard, though 
    # if Net Force is 0, the origin doesn't technically matter.
    com = test_atoms.get_center_of_mass()
    rel_positions = positions - com
    
    # 6. Calculate individual torques (r x F)
    torques = np.cross(rel_positions, forces)
    
    # 7. Sum to get Net Torque
    net_torque = np.sum(torques, axis=0)
    
    print(f"Net Torque (x, y, z): {net_torque}")
    print(f"Magnitude: {np.linalg.norm(net_torque)}")
    
    # Check against tolerance (usually ~1e-5 or 1e-6 for float32)
    if np.linalg.norm(net_torque) < 1e-4:
        print("PASS: Angular momentum is conserved (Torque is zero).")
    else:
        print("FAIL: Significant torque detected.")

    del test_atoms


