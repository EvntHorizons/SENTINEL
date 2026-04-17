"""
Astrodynamic constants used throughout SENTINEL.

Sources
-------
Earth parameters: WGS84 (NIMA TR8350.2, 2000)
Solar system: IAU 2012 / DE440
"""
import numpy as np
# Physical Parameters
C_LIGHT = 299792.458  # km/s (Speed of light in vacuum)
G0 = 9.80665e-3 # km/s^2 (Standard gravity for Isp calculations)

# Earth
MU_EARTH = 398600.4418 # km^3/s^2  (gravitational parameter)
R_EARTH = 6378.137 # km (WGS84 equatorial radius)
R_EARTH_MEAN = 6371.0088  # km (IUGG mean Earth radius)
J2_EARTH = 1.08262668e-3 # (unnormalized zonal harmonic)
J3_EARTH = -2.5326564e-6
J4_EARTH = -1.61962159e-6
OMEGA_EARTH = np.array([0.0, 0.0, 7.2921150e-5]) # rad/s (Earth rotation rate)
OMEGA_EARTH_SCALAR = 7.2921150e-5 # rad/s
F_EARTH = 1.0 / 298.257223563  # WGS84 flattening
E2_EARTH = 2 * F_EARTH - F_EARTH**2 # first eccentricity squared

# Sun
MU_SUN = 132712440018.0# km^3/s^2
AU_KM = 149597870.7 # km (1 AU)

# Moon
MU_MOON = 4902.800066 # km^3/s^2
# Cislunar Parameters (CR3BP)
# Mass ratio parameter: mu* = m2 / (m1 + m2)
MU_STAR_EM = MU_MOON / (MU_EARTH + MU_MOON) 
LU_EM = 384400.0  # km (Characteristic Length: Mean Earth-Moon distance)

# Solar Radiation Pressure 
P_SR = 4.56e-6 # N/m^2 at 1 AU

# Earth Atmosphere (Standard Exponential Model Reference)
RHO0_EARTH = 1.225e9  # kg/km^3 (Standard sea-level density)
H0_EARTH = 8.5  # km (Approximate scale height for troposphere)