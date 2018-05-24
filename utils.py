"""
Utilties

"""

import numpy
from scipy import interpolate


def stretch(gamma):
    lut = (gamma * numpy.arange(65536)).astype(numpy.uint)
    lut[lut > 254] = 254
    return lut


def calc_gamma(avg_int):
    return 2.7 if avg_int == 0.0 else 29.378 * avg_int ** -0.86


def interp_array(a, size=25):
    x, y = numpy.mgrid[-1:1:9j, -1:1:9j]
    z = a
    xnew, ynew = numpy.mgrid[-1:1:size * j, -1:1:size * j]
    tck = interpolate.bisplrep(x, y, z, s=0)
    znew = interpolate.bisplev(xnew[:, 0], ynew[0, :], tck)
    return znew
