"""Core astrodynamics engine — propagators, perturbations, coordinate frames, and time."""

from .constants import (
    MU_EARTH, R_EARTH, J2_EARTH, J3_EARTH, J4_EARTH,
    OMEGA_EARTH, OMEGA_EARTH_SCALAR, F_EARTH, E2_EARTH,
    MU_SUN, AU_KM, MU_MOON, P_SR,
)
from .orbital_elem_conv import (
    cartesian_to_classical, classical_to_cartesian,
    classical_to_equinoctial, equinoctial_to_classical,
    true_anomaly_to_mean, mean_anomaly_to_true,
    true_to_eccentric, eccentric_to_true,
    eccentric_to_mean, mean_to_eccentric,
    true_to_hyperbolic, hyperbolic_to_true,
    hyperbolic_to_mean, mean_to_hyperbolic,
)
from .propagator import propagate_keplerian
from .numerical_propagator import propagate_numerical, get_force_models
from .perturbations import (
    j2_acceleration, j3_acceleration, j4_acceleration,
    analytical_j2_drift,
)
from .coord_frame_transform import (
    eci_to_ecef, ecef_to_eci,
    eci_to_lvlh, lvlh_to_eci,
    eci_to_perifocal, perifocal_to_eci,
    geodetic_to_ecef, ecef_to_enu, enu_to_ecef,
)
from .drag import drag_acceleration, exponential_atmosphere_density
from .srp import srp_acceleration, cylindrical_shadow_factor
from .third_body import (
    sun_acceleration, moon_acceleration,
    get_sun_pos, get_moon_pos,
)
from .time_utils import Time, ensure_time, gmst, elapsed_sec, epoch_range
