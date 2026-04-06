import numpy as np

def calculate_delta_v(v_init: np.ndarray, v_final: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the total Delta-V for a transfer.
    
    Parameters
    ----------
    v_init : np.ndarray
        Initial velocity of the spacecraft before transfer
    v_final : np.ndarray
        Final target velocity at arrival
    v1 : np.ndarray
        Required transfer departure velocity
    v2 : np.ndarray
        Required transfer arrival velocity
        
    Returns
    -------
    float
        Total Delta-V (magnitude)
    """
    dv1 = np.linalg.norm(v1 - v_init)
    dv2 = np.linalg.norm(v_final - v2)
    return float(dv1 + dv2)

def lambert_izzo(mu: float, r1: np.ndarray, r2: np.ndarray, tof: float, prograde: bool = True, max_revs: int = 0):
    """
    Solves Lambert's problem using Izzo's (2014) algorithm with Householder iterations.
    
    Parameters
    ----------
    mu : float
        Gravitational parameter
    r1 : np.ndarray
        Initial position vector
    r2 : np.ndarray
        Final position vector
    tof : float
        Time of flight
    prograde : bool
        If True, transfer orbit is prograde (short way if < 180). Default True.
    max_revs : int
        Maximum number of full revolutions. If > 0, returns multiple branches.
        
    Returns
    -------
    v1 : list of np.ndarray
        List of departure velocity vectors for all valid solution branches
    v2 : list of np.ndarray
        List of arrival velocity vectors for all valid solution branches
    """
    from poliastro.iod import izzo
    
    v1_sols = []
    v2_sols = []
    
    for M in range(max_revs + 1):
        try:
            for v1_branch, v2_branch in izzo._lambert(mu, r1, r2, tof, M, 35, 1e-8):
                v1_sols.append(np.array(v1_branch))
                v2_sols.append(np.array(v2_branch))
        except ValueError:
            pass
            
    return v1_sols, v2_sols
