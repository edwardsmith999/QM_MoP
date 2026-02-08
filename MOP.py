import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt

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


def get_MOP_kinetic(r, r_prev, mv, E, Lz, Nplanes):
    """
    MOP kinetic calculation 
    P^k(t) = \sum_i \boldsymbol{v}_{i} (t) (sgn(z_p - z_i(t+dt)) - sgn(z_p - z_i(t)))
    """

    Nbins = Nplanes - 1
    dz = Lz / Nbins
    z_planes = np.arange(Nplanes)*dz 

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
                MOPstress_k[b] += 0.5 * mv[i] * cross
                MOPenergy_k[b] += 0.5 * E[i]  * cross 


    return MOPstress_k, MOPenergy_k 

#@njit(fastmath=True, cache=True)
@optional_njit(fastmath=True, cache=True)
def get_MOP_stress_power(r_z, fij, fijvi, Lz, Nplanes, threshold=1e-7):
    """
    Simple Numba version 
    Returns MOPstress_c array
    MOP Configuratitonal calculation 
    P^c = \sum_i \sum_j \boldsymbol{f}_{ij} (sgn(z_p - zi) - sgn(z_p - zj))
    """
    Nbins = Nplanes - 1
    n_atoms = r_z.shape[0]
    n_dims = fij.shape[2]
    MOPstress_c = np.zeros((Nplanes, n_dims)) 
    #Add power for energy calculation
    MOPpower_c = np.zeros((Nplanes,1)) 
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




def reconstruct_energy_flux(E_hist, E_k, dt, boundary_condition='zero'):
    """
    Reconstruct energy flux E_c on each plane from energy conservation.
    
    For each control volume I at each time step:
    dedt[t] - Eds_k[t+1]/dt = Eds_c[t]
    where Eds_c[t] = E_c[t, I+1] - E_c[t, I]
    
    With boundary conditions: E_c[:, 0] = E_c[:, Nplanes-1] = 0 (open boundaries)
    
    Parameters:
    -----------
    E_hist : array, shape (Nsteps+1, Nbins)
        Energy in each bin over time
    E_k : array, shape (Nsteps+1, Nplanes)
        Advection energy flux (size (Nsteps+1) x Nplanes)
    dt : float
        Time step
    boundary_condition : str
        'zero' for open boundaries (default)
    
    Returns:
    --------
    E_c : array, shape (Nsteps+1, Nplanes, ...)
        Reconstructed configurational energy flux on each plane
    """
    
    Nsteps_plus_one = E_hist.shape[0]
    Nsteps = Nsteps_plus_one - 1
    Nbins = E_hist.shape[1]
    Nplanes = Nbins + 1
    
    # Initialize E_c array with same shape as E_k
    E_c = np.zeros_like(E_k)
    
    # Calculate de/dt for each bin
    dedt = np.diff(E_hist, axis=0) / dt  # Shape: (Nsteps, Nbins)
    
    # For each time step t, we need to find E_c[t, :] such that:
    # dedt[t, i] - (E_k[t+1, i+1] - E_k[t+1, i])/dt = E_c[t, i+1] - E_c[t, i]
    
    for t in range(Nsteps):
        # Calculate Eds_k at time t+1 for each bin
        Eds_k_tp1 = E_k[t+1, 1:] - E_k[t+1, :-1]  # Shape: (Nbins, ...)
        
        # Right hand side for each bin: dedt[t, i] - Eds_k[t+1, i]/dt
        rhs = dedt[t, :] - Eds_k_tp1 / dt  # Shape: (Nbins,)
        
        if boundary_condition == 'zero':
            # Open boundaries: E_c[t, 0] = 0, E_c[t, Nplanes-1] = 0
            # From conservation: E_c[t, i+1] - E_c[t, i] = rhs[i]
            # Therefore: E_c[t, i+1] = E_c[t, i] + rhs[i]
            
            # Starting from E_c[t, 0] = 0, accumulate
            E_c[t, 0] = 0.0
            for i in range(Nbins):
                E_c[t, i+1] = E_c[t, i] + rhs[i]
            
            # Check top boundary residual
            top_residual = E_c[t, -1]
            
            # Distribute residual linearly to satisfy both boundary conditions
            # This makes the system slightly overdetermined, so we find the
            # least-squares solution that satisfies both boundaries
            if abs(top_residual) > 1e-10:
                correction = np.linspace(0, -top_residual, Nplanes)
                E_c[t, :] += correction
    
    # Handle last time step (Nsteps) - set to zero or copy from previous
    E_c[Nsteps, :] = 0.0  # or could use E_c[Nsteps-1, :]
    
    return E_c



def reconstruct_energy_flux_ref(E_hist, E_k, dt, reference=0.0, periodic=True):
    """
    Reconstruct energy flux E_c on each plane from energy conservation.
    
    For each control volume I at each time step:
    dedt[t] - Eds_k[t+1]/dt = Eds_c[t]
    where Eds_c[t] = E_c[t, I+1] - E_c[t, I]
    
    With boundary conditions: E_c[:, 0] = E_c[:, Nplanes-1] = 0 (open boundaries)
    
    Parameters:
    -----------
    E_hist : array, shape (Nsteps+1, Nbins)
        Energy in each bin over time
    E_k : array, shape (Nsteps+1, Nplanes)
        Advection energy flux (size (Nsteps+1) x Nplanes)
    dt : float
        Time step
    
    Returns:
    --------
    E_c : array, shape (Nsteps+1, Nplanes, ...)
        Reconstructed configurational energy flux on each plane
    """
    
    Nsteps_plus_one = E_hist.shape[0]
    Nsteps = Nsteps_plus_one - 1
    Nbins = E_hist.shape[1]
    Nplanes = Nbins + 1
    
    # Initialize E_c array with same shape as E_k
    E_c = np.zeros_like(E_k)
    
    # Calculate de/dt for each bin
    dedt = np.diff(E_hist, axis=0) / dt  # Shape: (Nsteps, Nbins)

    #Check references is either single value or array per time
    if type(reference) == float or type(reference) == int:
        reference = reference*np.ones(Nsteps)
    elif type(reference) == np.ndarray:
        print(reference.shape[0], Nsteps)
        assert reference.shape[0]-1 == Nsteps
    
    # For each time step t, we need to find E_c[t, :] such that:
    # dedt[t, i] - (E_k[t+1, i+1] - E_k[t+1, i])/dt = E_c[t, i+1] - E_c[t, i]
    
    for t in range(Nsteps):
        # Calculate Eds_k at time t+1 for each bin
        Eds_k_tp1 = E_k[t+1, 1:] - E_k[t+1, :-1]  # Shape: (Nbins, ...)
        
        # Right hand side for each bin: dedt[t, i] - Eds_k[t+1, i]/dt
        rhs = dedt[t, :] - Eds_k_tp1 / dt  # Shape: (Nbins,)
        
        # Open boundaries: E_c[t, 0] = 0, E_c[t, Nplanes-1] = 0
        # From conservation: E_c[t, i+1] - E_c[t, i] = rhs[i]
        # Therefore: E_c[t, i+1] = E_c[t, i] + rhs[i]
        
        # Starting from E_c[t, 0] = reference, accumulate
        E_c[t, 0] = reference[t]
        for i in range(Nbins):
            E_c[t, i+1] = E_c[t, i] + rhs[i]
        
        if periodic:
            # Check top boundary residual (assume periodic and loops back)
            top_residual = E_c[t, -1] - reference[t]

            # Distribute residual linearly to satisfy both boundary conditions
            # This makes the system slightly overdetermined, so we find the
            # least-squares solution that satisfies both boundaries
            if abs(top_residual) > 1e-10:
                correction = np.linspace(0, -top_residual, Nplanes)
                E_c[t, :] += correction
        
    return E_c



def get_CV_terms(E_hist, E_k, E_c_reconstructed, dt, binno=0):
    """
    Verify that the reconstructed E_c satisfies energy conservation.
    
    Returns:
    --------
    conservation : array
        Should be EXACTLY zero (within machine precision) if reconstruction is correct
    """
    # Calculate terms exactly as in original code
    Eds_c = E_c_reconstructed[:, binno+1] - E_c_reconstructed[:, binno]
    Eds_k = E_k[:, binno+1] - E_k[:, binno]
    dedt = np.diff(E_hist[:, binno]) / dt
    
    # Conservation check: dedt[t] - Eds_c[t] - Eds_k[t+1]/dt
    # Align indices exactly as in original
    conservation = dedt[:] - Eds_c[:-1] - Eds_k[1:] / dt
    
    return conservation, Eds_c, Eds_k, dedt


def reconstruct_and_plot(E_hist, E_k, dt, plottype="Energy", binno=100, ref=None):
    """
    Complete reconstruction with visualization.
    """
    
    # Reconstruct E_c
    E_c_reconstructed = reconstruct_energy_flux(E_hist, E_k, dt)
    
    # Verify for a specific bin
    conservation, Eds_c, Eds_k, dedt = get_CV_terms(
        E_hist, E_k, E_c_reconstructed, dt, binno
    )
    
    # Calculate max error
    max_error = np.max(np.abs(conservation))
    print(f"Maximum conservation error: {max_error}")
    print(f"This should be ~1e-15 (machine precision)")
    
    # Plot
    fig, axs = plt.subplots(2, 1)
    
    # Bottom panel: Conservation check
    axs[0].plot(Eds_c[:-1], '--', label="$f_{ij} v_i$ (reconstructed)", zorder=4)
    if np.any(ref != None):
        Eds_c_ref = ref[:, binno+1] - ref[:, binno]
        axs[0].plot(Eds_c_ref[:], ':', label=r"$f_{ij} v_i$ (reference)", zorder=6)
    
    axs[0].plot(Eds_k[1:]/dt, '-o', label="$e_i v_i$", alpha=0.8)
    axs[0].plot(dedt[:], label=r"$\frac{d}{dt} \rho e_i$")

    axs[0].plot(conservation, "k", lw=0.5, label=f"Sum (max error: {max_error:.2e})")
    axs[0].set_ylabel(plottype + "Rate")
    axs[0].set_xlabel("Time Step")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(Eds_c[:-1], '--', label="$f_{ij} v_i$ (reconstructed)", zorder=4)
    if np.any(ref != None):
        Eds_c_ref = ref[:, binno+1] - ref[:, binno]
        axs[1].plot(Eds_c_ref[:], ':', label=r"$f_{ij} v_i$ (reference)", zorder=6)
    axs[1].plot(dedt[:]-Eds_k[1:]/dt, label=r"$\frac{d}{dt} \rho e_i - e_i v_i$")
    axs[1].plot(conservation, "k", lw=0.5, label=f"Sum (max error: {max_error:.2e})")
    axs[1].set_ylabel(plottype + " Rate")
    axs[1].set_xlabel("Time Step")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return E_c_reconstructed, conservation




