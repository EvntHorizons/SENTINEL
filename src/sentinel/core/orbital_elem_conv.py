import numpy as np

def cartesian_to_classical(r_vec, v_vec, mu=398600.4418):
    #r_vec & v_vec in ECI
    r = np.linalg.norm(r_vec) # distance [km]
    v = np.linalg.norm(v_vec) # speed [km/s]

    h_vec = np.cross(r_vec, v_vec) # specific angular momentum [km^3/s^2]
    h = np.linalg.norm(h_vec) # specific angular momentum mag.

    k = [0, 0, 1]
    n_vec = np.cross(k, h_vec)
    n = np.linalg.norm(n_vec)

    e_vec = (1/mu) * ((v**2 - mu/r) * r_vec - np.dot(r_vec, v_vec) * v_vec)
    e = np.linalg.norm(e_vec)

    energy = v**2 / 2 - mu / r

    if abs(energy) < 1e-10:
    # Parabolic (e ≈ 1): semi-major axis is infinite
        a = float('inf')
    elif energy < 0:
    # Elliptic: bound orbit, a > 0
        a = -mu / (2 * energy)
    else:
    # Hyperbolic: escape orbit, a < 0
        a = -mu / (2 * energy)

    #inclination angle
    i = np.arccos(np.clip(h_vec[2] / h, -1 ,1))

    if n > 1e-10:
        raan = np.arccos(np.clip(n_vec[0] / n, -1, 1))
        if n_vec[1] < 0:
            raan = 2 * np.pi - raan
    else:
        raan = 0.0 #equatorial

    # arg periapsis
    if n > 1e-10 and e > 1e-10: # nonequatorial noncirc
        omega = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1, 1))
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega
    elif e > 1e-10: #equatorial, eccentric
        omega = np.arctan2(e_vec[1], e_vec[0])
        if omega < 0:
            omega += 2* np.pi
    else:
        omega = 0.0 # circ

    #true anomaly
    if e > 1e-10:  # eccentric orbit
        nu = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1, 1))
        if np.dot(r_vec, v_vec) < 0:
            nu = 2 * np.pi - nu             # quadrant check: receding = past perigee
    else:  # circular
        if n > 1e-10:
            nu = np.arccos(np.clip(np.dot(n_vec, r_vec) / (n * r), -1, 1))
            if r_vec[2] < 0:
                nu = 2 * np.pi - nu
        else:  # circular equatorial
            nu = np.arctan2(r_vec[1], r_vec[0])
            if nu < 0:
                nu += 2 * np.pi

    return a, e, i, raan, omega, nu

def classical_to_cartesian(a, e, i, raan, omega, nu, mu = 398600.4418):
    # perifocal frame denoted as 'pqw'
    l = a * (1 - e**2) # semi latus rectum
    r = l / (1 + e * np.cos(nu))

    r_pqw = np.array([r * np.cos(nu), r*np.sin(nu), 0.0])
    v_pqw = np.sqrt(mu / l) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    # rotation matrix from pqw to eci
    # R = R3(-raan) * R1(-i) * R3(-omega)
    R = _pqw_to_eci_matrix(raan, i, omega)

    r_eci = R @ r_pqw
    v_eci = R @ v_pqw

    return r_eci, v_eci


def _pqw_to_eci_matrix(raan, i, omega):
    cO = np.cos(raan); sO = np.sin(raan)
    ci = np.cos(i);     si = np.sin(i)
    cw = np.cos(omega); sw = np.sin(omega)

    return np.array([
        [cO*cw - sO*sw*ci, -cO*sw - sO*cw*ci,  sO*si],
        [sO*cw + cO*sw*ci, -sO*sw + cO*cw*ci, -cO*si],
        [sw*si,             cw*si,              ci   ]
    ])


def classical_to_equinoctial(a, e, i, raan, omega, nu):
    p = a * (1 - e**2)
    f = e * np.cos(omega + raan)
    g = e * np.sin(omega + raan)
    h = np.tan(i / 2) * np.cos(raan)
    k = np.tan(i / 2) * np.sin(raan)
    L = raan + omega + nu
    return p, f, g, h, k, L


def equinoctial_to_classical(p, f, g, h, k, L):
    e = np.sqrt(f**2 + g**2)
    i = 2 * np.arctan(np.sqrt(h**2 + k**2))
    raan = np.arctan2(k, h)
    omega_plus_raan = np.arctan2(g, f)
    omega = omega_plus_raan - raan
    nu = L - omega_plus_raan
    a = p / (1 - e**2)
    # Normalize angles to [0, 2π)
    raan = raan % (2 * np.pi)
    omega = omega % (2 * np.pi)
    nu = nu % (2 * np.pi)
    return a, e, i, raan, omega, nu

#anomaly conversions
def true_anomaly_to_mean(nu, e):
    if e < 1 - 1e-10:  # elliptic
        E = true_to_eccentric(nu, e)
        return eccentric_to_mean(E, e)
    elif e > 1 + 1e-10:  # hyperbolic
        H = true_to_hyperbolic(nu, e)
        return hyperbolic_to_mean(H, e)
    else:
        # Near-parabolic, using Barker's equation
        # M_p = tan(nu/2) + (1/3)*tan(nu/2)^3
        D = np.tan(nu / 2)
        return D + D**3 / 3


def mean_anomaly_to_true(M, e):
    if e < 1 - 1e-10:  # elliptic
        E = mean_to_eccentric(M, e)
        return eccentric_to_true(E, e)
    elif e > 1 + 1e-10:  # hyperbolic
        H = mean_to_hyperbolic(M, e)
        return hyperbolic_to_true(H, e)
    else:
        # Near-parabolic: solving Barker's equation
        # M = D + D^3/3 where D = tan(nu/2)
        # Cubic in D, now using closed-form or Newton's method
        D = _solve_barker(M)
        return 2 * np.arctan(D)


def true_to_eccentric(nu, e):
    E = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2),
                        np.sqrt(1 + e) * np.cos(nu / 2))
    return E % (2 * np.pi)


def eccentric_to_mean(E, e):
    return E - e * np.sin(E)


def mean_to_eccentric(M, e, tol=1e-14, max_iter=50):
    # Newton-Raphson, Keplers equation
    E = M + e * np.sin(M)  # initial guess (good for e < 0.8)
    for _ in range(max_iter):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= dE
        if abs(dE) < tol:
            break
    return E


def eccentric_to_true(E, e):
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    return nu % (2 * np.pi)

def true_to_hyperbolic(nu, e):
    # tanh(H/2) = sqrt((e-1)/(e+1)) * tan(nu/2)
    half_tan = np.sqrt((e - 1) / (e + 1)) * np.tan(nu / 2)
    H = 2 * np.arctanh(half_tan)
    return H


def hyperbolic_to_true(H, e):
    # tan(nu/2) = sqrt((e+1)/(e-1)) * tanh(H/2)
    nu = 2 * np.arctan(np.sqrt((e + 1) / (e - 1)) * np.tanh(H / 2))
    return nu


def hyperbolic_to_mean(H, e):
    # M_h = e * sinh(H) - H
    return e * np.sinh(H) - H


def mean_to_hyperbolic(M_h, e, tol=1e-14, max_iter=50):
    # Initial guess: H = M_h works for small M_h,
    # but for large M_h, sinh grows exponentially so
    # a better guess is H = sign(M_h) * ln(2|M_h|/e)
    if abs(M_h) < 1.0:
        H = M_h
    else:
        H = np.sign(M_h) * np.log(2 * abs(M_h) / e)

    for _ in range(max_iter):
        f = e * np.sinh(H) - H - M_h
        fp = e * np.cosh(H) - 1         # derivative
        dH = f / fp
        H -= dH
        if abs(dH) < tol:
            break

    return H


def _solve_barker(M):
    # Solve Barker's equation: M = D + D^3/3, where D = tan(nu/2)
    # Newton-Raphson on f(D) = D + D^3/3 - M
    D = M  # initial guess
    for _ in range(50):
        f = D + D**3 / 3 - M
        fp = 1 + D**2
        dD = f / fp
        D -= dD
        if abs(dD) < 1e-15:
            break
    return D
