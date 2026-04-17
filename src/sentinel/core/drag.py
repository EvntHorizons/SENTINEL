"""
Atmospheric drag perturbation model.

Uses a piecewise-exponential atmosphere density model with 25 altitude
bands. Density is returned in kg/km^3 for unit consistency with the
force model pipeline (positions in km, velocities in km/s).
"""
import numpy as np
from sentinel.core.constants import R_EARTH, OMEGA_EARTH, F_EARTH

# Altitude bands: (h0 [km], rho0 [kg/km^3], H [km])
ATMOSPHERE_BANDS = np.array([
    [0.0, 1.225e9, 7.249],
    [25.0, 3.899e7, 6.349],
    [30.0, 1.774e7, 6.682],
    [40.0, 3.972e6, 7.554],
    [50.0, 1.057e6, 8.382],
    [60.0, 3.206e5, 7.714],
    [70.0, 8.770e4, 6.549],
    [80.0, 1.905e4, 5.799],
    [90.0, 3.396e3, 5.382],
    [100.0, 5.297e2, 5.877],
    [110.0, 9.661e1, 7.263],
    [120.0, 2.438e1, 9.473],
    [130.0, 8.484, 12.636],
    [140.0, 3.845, 16.149],
    [150.0, 2.070, 22.523],
    [200.0, 2.541e-1, 37.105],
    [250.0, 6.073e-2, 45.546],
    [300.0, 1.916e-2, 53.628],
    [400.0, 2.803e-3, 58.515],
    [500.0, 5.245e-4, 60.828],
    [600.0, 1.138e-4, 69.752],
    [700.0, 2.846e-5, 82.593],
    [800.0, 8.163e-6, 97.411],
    [900.0, 2.651e-6, 118.411],
    [1000.0, 9.680e-7, 144.595]
], dtype=np.float64)

# Pre-slice arrays for vectorized lookup efficiency
_H0_ARR = ATMOSPHERE_BANDS[:, 0]
_RHO0_ARR = ATMOSPHERE_BANDS[:, 1]
_H_ARR = ATMOSPHERE_BANDS[:, 2]


def exponential_atmosphere_density(alt_km):
    alt = np.atleast_1d(np.asarray(alt_km, dtype=np.float64))
    alt = np.clip(alt, 0.0, None)  # Prevent negative altitudes

    # Vectorized bin lookup using binary search
    idx = np.searchsorted(_H0_ARR, alt, side='right') - 1
    idx = np.clip(idx, 0, len(_H0_ARR) - 1)

    h0 = _H0_ARR[idx]
    rho0 = _RHO0_ARR[idx]
    H = _H_ARR[idx]

    rho = rho0 * np.exp(-(alt - h0) / H)
     
    return rho if np.ndim(alt_km) > 0 else rho.item()


def drag_acceleration(t, r, v, mu, Cd, area_m2, mass_kg):
    # Enforce shape and precision
    r = np.atleast_2d(np.asarray(r, dtype=np.float64))
    v = np.atleast_2d(np.asarray(v, dtype=np.float64))

    r_mag = np.linalg.norm(r, axis=-1, keepdims=True)
    
    # Fast latitude approximation for oblate Earth radius
    sin_phi = r[..., 2:3] / r_mag
    r_eff = R_EARTH * (1.0 - F_EARTH * sin_phi**2)
    
    alt = r_mag - r_eff

    # Initialize acceleration array
    accel = np.zeros_like(r)

    # Mask for states actively experiencing drag
    active_mask = (alt < 1500.0).flatten()

    if np.any(active_mask):
        r_active = r[active_mask]
        v_active = v[active_mask]
        alt_active = alt[active_mask].flatten()

        rho = exponential_atmosphere_density(alt_active)[:, np.newaxis]

        # Velocity relative to co-rotating atmosphere
        v_atm = np.cross(OMEGA_EARTH, r_active, axis=-1)
        v_rel = v_active - v_atm
        v_rel_mag = np.linalg.norm(v_rel, axis=-1, keepdims=True)

        area_km2 = area_m2 * 1e-6

        # Calculate drag only for active states
        accel[active_mask] = -0.5 * rho * (Cd * area_km2 / mass_kg) * v_rel_mag * v_rel

    return accel.squeeze()