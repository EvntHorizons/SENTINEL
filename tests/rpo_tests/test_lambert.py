import numpy as np
import pytest
from poliastro.iod import izzo
from poliastro import constants

from sentinel.rpo.lambert import lambert_izzo, calculate_delta_v

def test_lambert_izzo_vallado_ex5_2():
    # Vallado 4th Ed, Example 5-2, p. 353
    mu = 3.986004418e5 # Earth GM in km^3/s^2
    
    r1 = np.array([5000.0, 10000.0, 2100.0]) # km
    r2 = np.array([-14600.0, 2500.0, 7000.0]) # km
    tof = 3600.0 # 1 hour
    
    v1_sols, v2_sols = lambert_izzo(mu, r1, r2, tof, prograde=True)
    
    # Vallado answer for single rev
    v1_expected = np.array([-5.992494, 1.925359, 3.245637])
    v2_expected = np.array([-3.312458, -4.196615, -0.385289])
    
    np.testing.assert_allclose(v1_sols[0], v1_expected, atol=1e-4)
    np.testing.assert_allclose(v2_sols[0], v2_expected, atol=1e-4)

def test_lambert_izzo_vs_poliastro():
    mu = 398600.4418
    r1 = np.array([5000.0, 10000.0, 2100.0])
    r2 = np.array([-14600.0, 2500.0, 7000.0])
    tof = 3600.0
    
    # Run our implementation
    v1_sols, v2_sols = lambert_izzo(mu, r1, r2, tof, prograde=True)
    
    # Run poliastro reference
    v1_pol_val, v2_pol_val = izzo.lambert(mu, r1, r2, tof)
    
    np.testing.assert_allclose(v1_sols[0], v1_pol_val, atol=1e-8)
    np.testing.assert_allclose(v2_sols[0], v2_pol_val, atol=1e-8)

def test_calculate_delta_v():
    v_init = np.array([1.0, 0.0, 0.0])
    v_final = np.array([0.0, 2.0, 0.0])
    v1 = np.array([1.5, 0.0, 0.0])
    v2 = np.array([0.0, 2.5, 0.0])
    
    dv = calculate_delta_v(v_init, v_final, v1, v2)
    np.testing.assert_allclose(dv, 1.0, atol=1e-10)
