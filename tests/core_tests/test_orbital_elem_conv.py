import pytest
import numpy as np
from sentinel.core.orbital_elem_conv import (
    cartesian_to_classical, classical_to_cartesian,
    classical_to_equinoctial, equinoctial_to_classical,
    true_to_eccentric, eccentric_to_mean, mean_to_eccentric,
    eccentric_to_true,
    true_to_hyperbolic, hyperbolic_to_true,
    hyperbolic_to_mean, mean_to_hyperbolic,
)

MU = 398600.4418


class TestRoundTrips:
    # Convert forward and back, verify nanometer precision

    # (label, r_eci, v_eci)
    CASES = [
        ("ISS-like LEO",
         np.array([6778.0, 0.0, 0.0]),
         np.array([0.0, 7.668, 0.0])),

        ("GEO (near-circular, near-equatorial)",
         np.array([42164.0, 0.0, 0.0]),
         np.array([0.0, 3.0747, 0.0])),

        ("Molniya (high eccentricity)",
         np.array([-6045.0, -3490.0, 2500.0]),
         np.array([-3.457, 6.618, 2.534])),

        ("Inclined elliptical",
         np.array([7000.0, 1000.0, 2000.0]),
         np.array([-1.5, 7.0, 1.0])),
    ]

    @pytest.mark.parametrize("label, r, v", CASES)
    def test_cartesian_classical_roundtrip(self, label, r, v):
        elems = cartesian_to_classical(r, v, MU)
        r2, v2 = classical_to_cartesian(*elems, MU)
        np.testing.assert_allclose(r2, r, atol=1e-10)
        np.testing.assert_allclose(v2, v, atol=1e-10)

    @pytest.mark.parametrize("label, r, v", CASES)
    def test_full_chain_roundtrip(self, label, r, v):
        # Cartesian -> Classical -> Equinoctial -> Classical -> Cartesian
        cla = cartesian_to_classical(r, v, MU)
        equ = classical_to_equinoctial(*cla)
        cla2 = equinoctial_to_classical(*equ)
        r2, v2 = classical_to_cartesian(*cla2, MU)
        np.testing.assert_allclose(r2, r, atol=1e-8)
        np.testing.assert_allclose(v2, v, atol=1e-8)


class TestKnownValues:
    # Verify against standard course/textbook values

    def test_circular_orbit_elements(self):
        # Circular equatorial orbit should have e≈0, i≈0
        r_geo = np.array([42164.0, 0.0, 0.0])
        v_geo = np.array([0.0, 3.0747, 0.0])
        a, e, i, raan, omega, nu = cartesian_to_classical(r_geo, v_geo, MU)
        assert abs(a - 42164.0) < 10.0 # semi-major axis near GEO
        assert e < 0.01 # near-circular
        assert i < 0.01 # near-equatorial

    def test_energy_conservation(self):
        """Semi-major axis should give correct orbital energy."""
        r = np.array([-6045.0, -3490.0, 2500.0])
        v = np.array([-3.457, 6.618, 2.534])
        a, *_ = cartesian_to_classical(r, v, MU)
        energy_from_rv = np.linalg.norm(v)**2 / 2 - MU / np.linalg.norm(r)
        energy_from_a = -MU / (2 * a)
        assert abs(energy_from_rv - energy_from_a) < 1e-10


class TestAnomalyConversions:

    def test_kepler_roundtrip(self):
        # M -> E -> nu -> E -> M should be identity
        for e in [0.0, 0.1, 0.5, 0.8, 0.99]:
            for M in np.linspace(0, 2*np.pi, 20):
                E = mean_to_eccentric(M, e)
                nu = eccentric_to_true(E, e)
                E2 = true_to_eccentric(nu, e)
                M2 = eccentric_to_mean(E2, e)
                assert abs(M2 - M) < 1e-12 or abs(abs(M2 - M) - 2*np.pi) < 1e-12

    def test_circular_anomalies_equal(self):
        """For e=0, M = E = nu."""
        for M in [0.0, 1.0, 3.14, 5.5]:
            E = mean_to_eccentric(M, 0.0)
            assert abs(E - M) < 1e-14

class TestHyperbolicAnomalies:

    @pytest.mark.parametrize("e", [1.5, 2.0, 5.0, 10.0])
    def test_hyperbolic_roundtrip(self, e):
        """M_h -> H -> nu -> H -> M_h should be identity."""
        for M_h in np.linspace(-5, 5, 20):
            H = mean_to_hyperbolic(M_h, e)
            nu = hyperbolic_to_true(H, e)
            H2 = true_to_hyperbolic(nu, e)
            M_h2 = hyperbolic_to_mean(H2, e)
            assert abs(M_h2 - M_h) < 1e-12

    def test_periapsis_is_zero(self):
        """At periapsis, nu = 0, H = 0, M_h = 0."""
        for e in [1.5, 3.0, 10.0]:
            assert abs(true_to_hyperbolic(0.0, e)) < 1e-15
            assert abs(hyperbolic_to_mean(0.0, e)) < 1e-15

    def test_true_anomaly_range(self):
        """Computed nu should always be within asymptote limit."""
        e = 2.0
        nu_max = np.arccos(-1/e)  # 120 degrees
        for M_h in np.linspace(-10, 10, 50):
            H = mean_to_hyperbolic(M_h, e)
            nu = hyperbolic_to_true(H, e)
            assert abs(nu) < nu_max


class TestHyperbolicCartesian:

    def test_hyperbolic_roundtrip_cartesian(self):
        """Hyperbolic state vector: convert to elements and back."""
        # Earth escape trajectory
        r = np.array([6578.0, 0.0, 0.0])          # 200 km altitude
        v = np.array([0.0, 12.0, 0.0])             # well above escape velocity

        a, e, i, raan, omega, nu = cartesian_to_classical(r, v, MU)

        assert a < 0, "Semi-major axis should be negative for hyperbolic"
        assert e > 1, "Eccentricity should be > 1 for hyperbolic"

        r2, v2 = classical_to_cartesian(a, e, i, raan, omega, nu, MU)
        np.testing.assert_allclose(r2, r, atol=1e-10)
        np.testing.assert_allclose(v2, v, atol=1e-10)

    def test_escape_velocity_boundary(self):
        """At exactly escape velocity, e should be ~1 and |a| very large."""
        r_mag = 6578.0  # 200 km altitude
        v_esc = np.sqrt(2 * MU / r_mag)  # escape velocity
        r = np.array([r_mag, 0.0, 0.0])
        v = np.array([0.0, v_esc, 0.0])

        a, e, *_ = cartesian_to_classical(r, v, MU)
        assert abs(e - 1.0) < 1e-6  # near-parabolic
