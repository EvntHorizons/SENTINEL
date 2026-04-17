"""
Coordinate frame transformations.

Supports ECI <-> ECEF, ECI <-> LVLH (RSW), ECI <-> Perifocal (PQW),
ECEF <-> ENU (topocentric), and Geodetic <-> ECEF.
"""
import numpy as np
import warnings
from astropy.time import Time
from astropy.coordinates import GCRS, ITRS, ICRF, CartesianRepresentation, CartesianDifferential, TEME
from astropy import units as u
from astropy.utils.iers import conf
from astropy.utils.iers import iers

# Attempt to update IERS, but fallback to cached data to prevent crashes when offline
try:
    conf.auto_download = True
    iers.IERS_Auto.open()
except Exception as e:
    warnings.warn(f"Could not connect to IERS servers. Relying on local/cached EOP data. Error: {e}")
    conf.auto_download = False

from sentinel.core.constants import OMEGA_EARTH, R_EARTH, F_EARTH, E2_EARTH
from sentinel.core.orbital_elem_conv import _pqw_to_eci_matrix


# ─ ECI <-> ECEF

def eci_to_ecef(r_eci, v_eci, time_utc, eci_frame='gcrs'):
    """
    Transform ECI state to ECEF (ITRS) using IAU 2000/2006 models.

    Accounts for Precession, Nutation, Earth Rotation, and Polar Motion 
    using IERS daily Earth Orientation Parameters.
    """
    if not isinstance(time_utc, Time):
        time_utc = Time(time_utc, scale='utc')

    # Convert inputs to float64 numpy arrays to ensure precision
    r_eci = np.asarray(r_eci, dtype=np.float64)
    v_eci = np.asarray(v_eci, dtype=np.float64)

    # Construct Astropy representations with explicit units
    pos_rep = CartesianRepresentation(r_eci * u.km)
    vel_rep = CartesianDifferential(v_eci * u.km / u.s)
    
    # Instantiate the correct source frame
    eci_frame_lower = eci_frame.lower()
    if eci_frame_lower == 'gcrs':
        source_state = GCRS(pos_rep.with_differentials(vel_rep), obstime=time_utc)
    elif eci_frame_lower == 'icrf':
        # ICRF is solar-system barycentric; Astropy handles the geocentric translation internally
        source_state = ICRF(pos_rep.with_differentials(vel_rep))
    elif eci_frame_lower == 'teme':
        source_state = TEME(pos_rep.with_differentials(vel_rep), obstime=time_utc)
    else:
        raise ValueError(f"Unsupported ECI frame: {eci_frame}. Choose 'gcrs', 'icrf', or 'teme'.")
    
    # Transform to ECEF (ITRS)
    ecef_state = source_state.transform_to(ITRS(obstime=time_utc))
    
    # Extract numpy arrays
    r_ecef = ecef_state.cartesian.xyz.to_value(u.km)
    v_ecef = ecef_state.cartesian.differentials['s'].d_xyz.to_value(u.km / u.s)
    
    return r_ecef, v_ecef

def ecef_to_eci(r_ecef, v_ecef, time_utc, eci_frame='gcrs'):
    """
    Transform ECEF (ITRS) state to ECI using high-fidelity IAU 2000/2006 models.

    Accounts for Precession, Nutation, Earth Rotation, and Polar Motion 
    using IERS daily Earth Orientation Parameters.
    """
    # Ensure time is an Astropy Time object
    if not isinstance(time_utc, Time):
        time_utc = Time(time_utc, scale='utc')

    # Convert inputs to float64 numpy arrays to ensure precision
    r_ecef = np.asarray(r_ecef, dtype=np.float64)
    v_ecef = np.asarray(v_ecef, dtype=np.float64)

    # Construct Astropy representations with explicit units
    pos_rep = CartesianRepresentation(r_ecef * u.km)
    vel_rep = CartesianDifferential(v_ecef * u.km / u.s)
    
    # Define the source ECEF (ITRS) state
    ecef_state = ITRS(pos_rep.with_differentials(vel_rep), obstime=time_utc)
    
    # Transform to the specified ECI frame
    eci_frame_lower = eci_frame.lower()
    if eci_frame_lower == 'gcrs':
        eci_state = ecef_state.transform_to(GCRS(obstime=time_utc))
    elif eci_frame_lower == 'icrf':
        # ICRF does not require an obstime as it is fixed relative to the barycenter
        eci_state = ecef_state.transform_to(ICRF())
    elif eci_frame_lower == 'teme':
        eci_state = ecef_state.transform_to(TEME(obstime=time_utc))
    else:
        raise ValueError(f"Unsupported ECI target frame: {eci_frame}. Choose 'gcrs', 'icrf', or 'teme'.")
    
    # Extract numpy arrays
    r_eci = eci_state.cartesian.xyz.to_value(u.km)
    v_eci = eci_state.cartesian.differentials['s'].d_xyz.to_value(u.km / u.s)
    
    return r_eci, v_eci

# ─ ECI <-> LVLH (RSW)

def eci_to_lvlh(r_tar, v_tar, r_ref, v_ref):
    """
    Convert target ECI state to LVLH (RSW) frame relative to reference.
    """
    # Enforce float64 to prevent catastrophic cancellation in close RPO
    r_tar = np.atleast_2d(np.asarray(r_tar, dtype=np.float64))
    v_tar = np.atleast_2d(np.asarray(v_tar, dtype=np.float64))
    r_ref = np.atleast_2d(np.asarray(r_ref, dtype=np.float64))
    v_ref = np.atleast_2d(np.asarray(v_ref, dtype=np.float64))

    # Reference orbit angular momentum
    h_ref = np.cross(r_ref, v_ref, axis=-1)
    
    # Magnitudes
    r_ref_mag = np.linalg.norm(r_ref, axis=-1, keepdims=True)
    h_ref_mag = np.linalg.norm(h_ref, axis=-1, keepdims=True)

    # Singularity safeguard
    if np.any(h_ref_mag < 1e-10):
        raise ValueError("Reference angular momentum is near zero (rectilinear orbit). LVLH frame is undefined.")

    # Unit vectors for LVLH (RSW)
    u_r = r_ref / r_ref_mag
    u_w = h_ref / h_ref_mag
    u_s = np.cross(u_w, u_r, axis=-1)

    # Construct Rotation Matrix [N, 3, 3] 
    # axis=-2 stacks the basis vectors as rows in each 3x3 matrix
    R = np.stack((u_r, u_s, u_w), axis=-2)

    # Relative ECI state
    r_rel_eci = r_tar - r_ref
    v_rel_eci = v_tar - v_ref

    # Angular velocity of the LVLH frame in ECI
    w_lvlh_eci = h_ref / (r_ref_mag**2)

    # Kinematic relative velocity adjustment (Transport Theorem)
    v_rel_eci_adj = v_rel_eci - np.cross(w_lvlh_eci, r_rel_eci, axis=-1)

    # Batched matrix-vector multiplication using Einstein summation
    # 'nij' (Rotation matrices N x 3 x 3) * 'nj' (Vectors N x 3) -> 'ni' (Result N x 3)
    r_lvlh = np.einsum('nij,nj->ni', R, r_rel_eci)
    v_lvlh = np.einsum('nij,nj->ni', R, v_rel_eci_adj)

    # Squeeze to return (3,) if original inputs were 1D, else return (N, 3)
    return r_lvlh.squeeze(), v_lvlh.squeeze()

def lvlh_to_eci(r_lvlh, v_lvlh, r_ref, v_ref):
    """
    Convert LVLH (RSW) relative state back to ECI.
    Supports single states (3,) or batched arrays of states (N, 3).
    """
    # Enforce float64 to prevent precision truncation during vector addition
    r_lvlh = np.atleast_2d(np.asarray(r_lvlh, dtype=np.float64))
    v_lvlh = np.atleast_2d(np.asarray(v_lvlh, dtype=np.float64))
    r_ref = np.atleast_2d(np.asarray(r_ref, dtype=np.float64))
    v_ref = np.atleast_2d(np.asarray(v_ref, dtype=np.float64))

    # Reference orbit angular momentum
    h_ref = np.cross(r_ref, v_ref, axis=-1)

    # Magnitudes
    r_ref_mag = np.linalg.norm(r_ref, axis=-1, keepdims=True)
    h_ref_mag = np.linalg.norm(h_ref, axis=-1, keepdims=True)

    # Singularity safeguard
    if np.any(h_ref_mag < 1e-10):
        raise ValueError("Reference angular momentum is near zero (rectilinear orbit). LVLH frame is undefined.")

    # Unit vectors for LVLH (RSW)
    u_r = r_ref / r_ref_mag
    u_w = h_ref / h_ref_mag
    u_s = np.cross(u_w, u_r, axis=-1)

    # Construct Rotation Matrix [N, 3, 3]
    R = np.stack((u_r, u_s, u_w), axis=-2)
    
    # Invert the rotation matrices by swapping ONLY the last two axes (the 3x3 matrices)
    R_inv = np.swapaxes(R, -1, -2)

    # Angular velocity of the LVLH frame in ECI
    w_lvlh_eci = h_ref / (r_ref_mag**2)

    # Convert relative position to ECI
    # 'nij' (R_inv N x 3 x 3) * 'nj' (Vectors N x 3) -> 'ni' (Result N x 3)
    r_rel_eci = np.einsum('nij,nj->ni', R_inv, r_lvlh)

    # Convert relative velocity to ECI (Transport Theorem)
    v_rel_eci_base = np.einsum('nij,nj->ni', R_inv, v_lvlh)
    v_rel_eci = v_rel_eci_base + np.cross(w_lvlh_eci, r_rel_eci, axis=-1)

    # Reconstruct the target's absolute ECI state
    r_tar = r_rel_eci + r_ref
    v_tar = v_rel_eci + v_ref

    # Squeeze to return (3,) if original inputs were 1D, else return (N, 3)
    return r_tar.squeeze(), v_tar.squeeze()

# ─ ECI <-> Perifocal 

def eci_to_perifocal(r_eci, v_eci, raan, i, omega):
    """
    Transform ECI state to perifocal (PQW) frame.
    Supports single states (3,) or batched arrays of states (N, 3).
    """
    # Enforce float64 for precision
    r_eci = np.atleast_2d(np.asarray(r_eci, dtype=np.float64))
    v_eci = np.atleast_2d(np.asarray(v_eci, dtype=np.float64))

    # Fetch the rotation matrix (assumes it returns [3, 3] or [N, 3, 3])
    R_pqw_to_eci = np.asarray(_pqw_to_eci_matrix(raan, i, omega), dtype=np.float64)

    # invert the rotation matrix by swapping only the last two axes
    R_eci_to_pqw = np.swapaxes(R_pqw_to_eci, -1, -2)

    # Universal matrix-vector multiplication using Einstein summation
    # The '...' handles either a missing batch dimension or an N-dimensional batch
    r_pqw = np.einsum('...ij,...j->...i', R_eci_to_pqw, r_eci)
    v_pqw = np.einsum('...ij,...j->...i', R_eci_to_pqw, v_eci)

    # Squeeze to return (3,) if original inputs were 1D, else return (N, 3)
    return r_pqw.squeeze(), v_pqw.squeeze()


def perifocal_to_eci(r_pqw, v_pqw, raan, i, omega):
    """
    Transform perifocal (PQW) state to ECI.
    Supports single states (3,) or batched arrays of states (N, 3).
    """
    # Enforce float64 for precision
    r_pqw = np.atleast_2d(np.asarray(r_pqw, dtype=np.float64))
    v_pqw = np.atleast_2d(np.asarray(v_pqw, dtype=np.float64))

    # Fetch the rotation matrix (returns [3, 3] or [N, 3, 3])
    R_pqw_to_eci = np.asarray(_pqw_to_eci_matrix(raan, i, omega), dtype=np.float64)

    # Universal matrix-vector multiplication using Einstein summation
    # The '...' handles either a missing batch dimension or an N-dimensional batch
    r_eci = np.einsum('...ij,...j->...i', R_pqw_to_eci, r_pqw)
    v_eci = np.einsum('...ij,...j->...i', R_pqw_to_eci, v_pqw)

    # Squeeze to return (3,) if original inputs were 1D, else return (N, 3)
    return r_eci.squeeze(), v_eci.squeeze()


# ─ Geodetic <-> ECEF 

def geodetic_to_ecef(lat_rad, lon_rad, h):
    """
    Convert geodetic coordinates to ECEF position.
    Supports single coordinates or batched arrays.
    """
    # Enforce float64 for precision
    lat_rad = np.asarray(lat_rad, dtype=np.float64)
    lon_rad = np.asarray(lon_rad, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    
    # Prime vertical radius of curvature
    N = R_EARTH / np.sqrt(1.0 - E2_EARTH * sin_lat**2)

    x = (N + h) * cos_lat * np.cos(lon_rad)
    y = (N + h) * cos_lat * np.sin(lon_rad)
    z = (N * (1.0 - E2_EARTH) + h) * sin_lat
    
    # Stack along the last axis to ensure (N, 3) shape for batched arrays
    # or (3,) for scalar inputs.
    r_ecef = np.stack((x, y, z), axis=-1)
    
    return r_ecef


def _ecef_to_enu_matrix(lat_rad, lon_rad):
    """
    Rotation matrix from ECEF to ENU (East-North-Up) frame.
    Supports single coordinates or batched arrays.
    """
    # Enforce float64 precision
    lat_rad = np.asarray(lat_rad, dtype=np.float64)
    lon_rad = np.asarray(lon_rad, dtype=np.float64)

    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
    
    # Create an array of zeros matching the input shape for the [0, 2] element
    zero = np.zeros_like(lat_rad)

    # Stack the elements along the last axis (-1) to form rows, 
    # then stack the rows along the second-to-last axis (-2) to form the matrices.
    row1 = np.stack([-sin_lon,           cos_lon,           zero], axis=-1)
    row2 = np.stack([-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat], axis=-1)
    row3 = np.stack([cos_lat*cos_lon,   cos_lat*sin_lon, sin_lat], axis=-1)
    
    R = np.stack([row1, row2, row3], axis=-2)
    
    return R

def ecef_to_enu(r_ecef, lat_rad, lon_rad, h):
    """
    Convert ECEF position to ENU coordinates relative to a ground station.
    Supports single coordinates (3,) or batched arrays (N, 3).
    """
    # Enforce float64 precision to prevent truncation during subtraction
    r_ecef = np.atleast_2d(np.asarray(r_ecef, dtype=np.float64))

    # Get station ECEF coordinates (inherits batch safety from previous refactor)
    r_station_ecef = geodetic_to_ecef(lat_rad, lon_rad, h)
    
    # Calculate relative vector in ECEF frame. 
    # NumPy broadcasting handles (N, 3) - (3,) or (N, 3) - (N, 3) natively.
    dr_ecef = r_ecef - r_station_ecef

    # Fetch rotation matrix
    R = _ecef_to_enu_matrix(lat_rad, lon_rad)
    
    # Universal matrix-vector multiplication using Einstein summation
    # Maps either a single 3x3 matrix to an N-length trajectory, 
    # or an N-batch of matrices to an N-batch of targets.
    r_enu = np.einsum('...ij,...j->...i', R, dr_ecef)
    
    # Squeeze to return (3,) if original inputs were 1D, else return (N, 3)
    return r_enu.squeeze()


def enu_to_ecef(r_enu, lat_rad, lon_rad, h):
    """
    Convert ENU coordinates to ECEF position.
    Supports single coordinates (3,) or batched arrays (N, 3).
    """
    # Enforce float64 precision to prevent truncation during addition
    r_enu = np.atleast_2d(np.asarray(r_enu, dtype=np.float64))

    # Get station ECEF coordinates
    r_station_ecef = geodetic_to_ecef(lat_rad, lon_rad, h)

    # Fetch the rotation matrix (ECEF to ENU)
    R = _ecef_to_enu_matrix(lat_rad, lon_rad)
    
    # Invert the rotation matrices BY swapping
    # to safely handle both (3, 3) and (N, 3, 3) shapes
    R_inv = np.swapaxes(R, -1, -2)
    
    # Universal matrix-vector multiplication using Einstein summation
    dr_ecef = np.einsum('...ij,...j->...i', R_inv, r_enu)
    
    # Reconstruct absolute ECEF state
    r_ecef = r_station_ecef + dr_ecef
    
    # Squeeze to return (3,) if original inputs were 1D, else return (N, 3)
    return r_ecef.squeeze()
