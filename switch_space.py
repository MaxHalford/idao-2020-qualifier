import numpy as np
import math
from collections import namedtuple

def pv_to_equin(x, y, z, Vx, Vy, Vz, mu = 398600441500000):
    """
    POS must be in meters
    //  Copyright (c) CNES  2008
    //
    //  This software is part of CelestLab, a CNES toolbox for Scilab
    //
    //  This software is governed by the CeCILL license under French law and
    //  abiding by the rules of distribution of free software.  You can  use,
    //  modify and/ or redistribute the software under the terms of the CeCILL
    //  license as circulated by CEA, CNRS and INRIA at the following URL
    //  'http://www.cecill.info'.
    """
    pos = np.array([x, y, z])
    vel = np.array([Vx, Vy, Vz])

    # Cl_norm function
    r = np.sqrt((pos ** 2).sum())
    V = np.sqrt((vel ** 2).sum())

    # Cl_cross function
    W = np.cross(pos, vel)

    a = r / (2 - r * V**2 / mu)

    esinE = np.dot(pos, vel) / np.sqrt(mu * a)

    Wu =  W / (W**2).sum()**0.5

    hx = - Wu[1] / (1 + Wu[2])

    hy = Wu[0] / (1 + Wu[2])

    h = 1 / (1 + hx**2 + hy**2)

    P = np.array([
        (1 + hx**2 - hy**2) * h,
         2 * hx * hy * h,
        -2 * hy * h
    ])

    Q = np.array([
        2 * hx * hy * h,
        (1 - hx**2 + hy**2) * h,
        2 * hx * h,
    ])

    X = np.dot(pos, P)
    Y = np.dot(pos, Q)
    VX = np.dot(vel, P)
    VY = np.dot(vel, Q)

    ex = np.sqrt((W ** 2).sum()) * VY / mu - X / r
    ey = -np.sqrt((W ** 2).sum()) * VX / mu - Y / r

    nu = 1 / (1 + np.sqrt(1 - ex**2 - ey**2))

    cosLE = X / a + ex - esinE * nu * ey

    sinLE = Y / a + ey + esinE * nu * ex

    L = math.atan2(sinLE,cosLE) - ex * sinLE + ey * cosLE

    nrev = np.floor(L / (2 * math.pi))
    y = L - nrev * (2 * math.pi)

    return a, ex, ey, hx, hy, y


# crbt Func
def cbrt(x):
    return np.sign(x) * np.abs(x)**(1/3)

def equin_to_pv(a, ex, ey, hx, hy, hz, mu = 398600441500000):
    """
    //  Copyright (c) CNES  2008
    //
    //  This software is part of CelestLab, a CNES toolbox for Scilab
    //
    //  This software is governed by the CeCILL license under French law and
    //  abiding by the rules of distribution of free software.  You can  use,
    //  modify and/ or redistribute the software under the terms of the CeCILL
    //  license as circulated by CEA, CNRS and INRIA at the following URL
    //  'http://www.cecill.info'.
    """
    # CL_kp_anomConvertCir Func
    # CL_kp_anomConvertCir(type_anom1,type_anom2,ex,ey,pso1)
    # F = CL_kp_anomConvertCir("M", "E", ex, ey, equin(6,:));
    pso1 = hz
    ecc = np.sqrt(ex**2 + ey**2)
    A = math.atan2(ey, ex)

    anom1 =  pso1 - A

    # Ellipse
    if ecc < 1:
        # CL__kp_M2Eell Func
        # CL_kp_M2E(e,M)
        # CL__kp_M2Eell(ecc(I), M(I))

        M = anom1
        e = ecc

        # CL_rMod Func
        # CL_rMod(x, a, b)
        # CL_rMod(M,-%pi,%pi)
        delta = math.pi - (- math.pi)
        nrev = np.floor((M - (- math.pi)) / delta)
        reduced_m = M - nrev * delta

        k1 = 3 * math.pi + 2
        k2 = math.pi - 1
        k3 = 6 * math.pi - 1
        A  = 3 * k2 * k2 / k1
        B  = k3 * k3 / (6 * k1)

        if abs(reduced_m) < (1./ 6.):
            E = reduced_m + e * (cbrt(6 * reduced_m) - reduced_m)

        if abs(reduced_m) >= (1./ 6.) and reduced_m < 0:
            w = math.pi + reduced_m
            E = reduced_m + e * (math.pi - A * w / (B - w) - reduced_m)

        if abs(reduced_m) >= (1./ 6.) and  reduced_m >= 0:
            w = math.pi - reduced_m
            E = reduced_m + e * (math.pi - A * w / (B - w) - reduced_m)

        e1 = 1 - e

        no_cancellation_risk = (e1 + E * E / 6) >= 0.1

        f = 0
        fd = 0

        for i in range(2):

            fdd = e * np.sin(E)
            fddd = e * np.cos(E)

            if no_cancellation_risk:

                f = (E - fdd) - reduced_m
                fd = 1 - fddd

            if not no_cancellation_risk:
                # CL__E_esinE Func
                # f(I)  = CL__E_esinE(e(I),E(I)) - reducedM(I);
                # CL__E_esinE(e,E)
                x = (1 - e) * np.sin(E)
                mE2 = -E * E
                term = E
                d = 0
                x0 = None
                iteration = 0
                nb_max_iter = 20
                K = True
                while K and iteration <= nb_max_iter:
                    d = d + 2
                    term = term * mE2 / (d * (d + 1))
                    x0 = x
                    x = x - term
                    # If x == x0: Stop
                    if x == x0:
                        K = False
                    iteration += 1
                # Output CL__E_esinE
                f = x

                s = np.sin(0.5 * E)
                fd = e1 + 2 * e * s * s

            dee = f * fd / (0.5 * f * fdd - fd * fd)
            w = fd + 0.5 * dee * (fdd + dee * fddd / 3)
            fd = fd + dee * (fdd + 0.5 * dee * fddd)
            E = E - (f - dee * (fd - w)) / fd

        E = E + (M - reduced_m)
    """
    # Hyperbola
    if ecc > 1:
        # CL__kp_M2Ehyp Func
        # E(I) = CL__kp_M2Ehyp(ecc(I), M(I));
        # CL__kp_M2Ehyp(ecc,M)

        E = math.asinh(M / ecc)

        if (ecc - 1)**2 + (0.5 * np.abs(M) - 1)**2 < 1.01:
            E = np.sign(M) * (6 * np.abs(M) / ecc) ** (1/3)

        nb_max_iter = 20
        errmax = 0.0000000000000222044605
        iteration = 0
        K = True
        while K and iteration <= nb_max_iter:
            E1 = E
            E = E - (ecc * np.sinh(E) - E - M) / (ecc * cosh(E) - 1)
            err1 = abs(E1 - E) / max(abs(E),1)
            err2 = abs(ecc * np.sinh(E) - E - M) / max(abs(M),1)
            if err1 <= errmax and err2 <= errmax:
                K = False
            iteration += 1

    """
    anom2 = E
    pso_2 = anom2 + math.atan2(ey, ex)

    F = pso_2

    cosF = np.cos(F)
    sinF = np.sin(F)

    r = a * (1 - ex * cosF - ey * sinF)
    n = np.sqrt(mu / a**3)
    nu = 1 / (1 + np.sqrt(1 - ex**2 - ey**2))

    X = a * ((1-nu * ey**2) * cosF + nu * ex * ey *sinF - ex)
    Y = a * ((1-nu * ex**2) * sinF + nu * ex * ey *cosF - ey)

    VX = n * a**2 / r * (cosF * nu * ex * ey + sinF * (nu * ey**2 - 1))
    VY = n * a**2 / r * (-sinF * nu * ex * ey - cosF * (nu * ex**2 - 1))

    h = 1 / (1 + hx**2 + hy**2)

    P = np.array([
        (1 + hx**2 - hy**2) * h,
        2 * hx * hy * h,
        -2 * hy * h
    ])

    Q = np.array([
        2 * hx * hy * h,
        (1 - hx**2 + hy**2) * h,
        2 * hx * h
    ])

    pos = P * X + Q * Y
    vel = P * VX + Q * VY

    return pos, vel
