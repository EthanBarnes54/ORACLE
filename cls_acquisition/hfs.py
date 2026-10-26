"""The hyperfine structure of one transition: where its lines fall and how strong each is.

Date:    2026-10-27
Version: 0.2.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The data viewer's estimator, without satlas: a spectrum drawn from a
nuclear spin and the two levels' A and B factors, to lay over a run and
judge by eye before the fit in DENIS. Each level of total angular
momentum J splits into F = |I - J| ... I + J, shifted from its centre by

    A K / 2 + B (3/4 K (K + 1) - I (I + 1) J (J + 1)) / (2 I (2 I - 1) J (2 J - 1)),

K = F (F + 1) - I (I + 1) - J (J + 1), the B term only where I and J are
both at least 1. A line joins F_l to F_u for F_u - F_l = -1, 0 or 1, but
not 0 to 0, with the Racah intensity

    (2 F_l + 1) (2 F_u + 1) {F_u F_l 1; J_l J_u I}^2,

so the lines from each lower level share its strength in proportion to
its 2 F + 1 states. Angular momenta are given as floats on the
half-integers.

Depends on: numpy.
"""

from math import factorial, sqrt

import numpy as np


class Hfs_Error(ValueError):
    pass


def _Doubled(value, name):
    doubled = round(2 * value)
    if abs(2 * value - doubled) > 1e-9 or doubled < 0:
        raise Hfs_Error(f"{name} is {value:g}; it must be a whole or half-whole number, not "
                        f"negative.")
    return doubled


def _Triangle(a, b, c):
    """The triangle coefficient of three doubled momenta, or None if they
    cannot couple."""
    if (a + b + c) % 2 or c < abs(a - b) or c > a + b:
        return None
    return sqrt(factorial((a + b - c) // 2) * factorial((a - b + c) // 2)
                * factorial((-a + b + c) // 2) / factorial((a + b + c) // 2 + 1))


def Wigner_6j(j1, j2, j3, j4, j5, j6):
    """{j1 j2 j3; j4 j5 j6} by Racah's formula; 0 where the momenta cannot
    couple."""
    d = [_Doubled(j, 'An angular momentum') for j in (j1, j2, j3, j4, j5, j6)]
    triads = ((d[0], d[1], d[2]), (d[0], d[4], d[5]), (d[3], d[1], d[5]), (d[3], d[4], d[2]))
    triangles = [_Triangle(*triad) for triad in triads]
    if None in triangles:
        return 0.0
    sums = [sum(triad) // 2 for triad in triads]
    pairs = ((d[0] + d[1] + d[3] + d[4]) // 2, (d[1] + d[2] + d[4] + d[5]) // 2,
             (d[2] + d[0] + d[5] + d[3]) // 2)
    total = 0.0
    for t in range(max(sums), min(pairs) + 1):
        denominator = 1
        for value in [t - s for s in sums] + [p - t for p in pairs]:
            denominator *= factorial(value)
        total += (-1) ** t * factorial(t + 1) / denominator
    return float(np.prod(triangles) * total)


def _F_Values(spin, j):
    return [abs(spin - j) + k for k in range(int(round(spin + j - abs(spin - j))) + 1)]


def Level_Shift(f, spin, j, a, b):
    """The hyperfine shift of level F from its centre, in the units of A and B."""
    k = f * (f + 1) - spin * (spin + 1) - j * (j + 1)
    shift = a * k / 2
    if spin >= 1 and j >= 1:
        shift += b * (0.75 * k * (k + 1) - spin * (spin + 1) * j * (j + 1)) / (
            2 * spin * (2 * spin - 1) * j * (2 * j - 1))
    return shift


def Lines(spin, j_lower, j_upper, a_lower, b_lower, a_upper, b_upper):
    """``[(F_l, F_u, position, intensity)]``: every allowed line, its position
    from the centroid in the units of the A and B factors and its Racah
    intensity, the strongest scaled to 1."""
    for value, name in ((spin, 'I'), (j_lower, 'J lower'), (j_upper, 'J upper')):
        _Doubled(value, name)
    if abs(j_upper - j_lower) > 1 or j_lower == j_upper == 0:
        raise Hfs_Error(f"J {j_lower:g} to {j_upper:g} is not an electric dipole transition.")
    lines = []
    for f_lower in _F_Values(spin, j_lower):
        for f_upper in _F_Values(spin, j_upper):
            if abs(f_upper - f_lower) > 1 or f_upper == f_lower == 0:
                continue
            intensity = (2 * f_lower + 1) * (2 * f_upper + 1) * Wigner_6j(
                f_upper, f_lower, 1, j_lower, j_upper, spin) ** 2
            if intensity > 1e-12:
                lines.append((f_lower, f_upper,
                              Level_Shift(f_upper, spin, j_upper, a_upper, b_upper)
                              - Level_Shift(f_lower, spin, j_lower, a_lower, b_lower),
                              intensity))
    strongest = max(line[3] for line in lines)
    return [(fl, fu, position, intensity / strongest) for fl, fu, position, intensity in lines]


def Spectrum(x, lines, centroid, fwhm, scale, background):
    """The model at ``x``: Lorentzians of full width ``fwhm`` at the lines'
    positions about ``centroid``, the strongest ``scale`` high, on a flat
    ``background``."""
    x = np.asarray(x, dtype=float)
    shape = np.zeros_like(x)
    for _, _, position, intensity in lines:
        shape += intensity / (1.0 + (2.0 * (x - centroid - position) / fwhm) ** 2)
    return background + scale * shape
