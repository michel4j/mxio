#!/usr/bin/env python

import imageio
import sys
from matplotlib import pyplot as plt
import numpy
from scipy import optimize, signal

XSTEP = 0.05

def reject_outliers(data, m = 2.):
    d = numpy.abs(data - numpy.median(data))
    mdev = numpy.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def calc_r(nx, ny, cx, cy, rotx, roty):
    x = (numpy.arange(nx) - int(cx))/numpy.cos(rotx)
    y = (numpy.arange(ny) - int(cy))/numpy.cos(roty)
    return numpy.hypot(x[:, None], y[None, :])


def calc_theta(nx, ny, cx, cy):
    x = (numpy.arange(nx) - int(cx))
    y = (numpy.arange(ny) - int(cy))
    return numpy.arctan2(2*x[:, None]/nx, 2*y[None, :]/ny)


class StdOptimizer(object):
    def __init__(self, data, nx, ny, mask, prec=1):
        self.mask = mask
        self.data = data
        self.nx, self.ny = nx, ny
        self.factors = 1.0/(prec*XSTEP)

    def __call__(self, coeffs, r_bin):
        cx, cy, rotx, roty = self.factors*coeffs

        rg = calc_r(self.nx, self.ny, cx, cy, rotx, roty).astype(numpy.int)
        work_r = rg[self.mask].ravel()

        intensities = numpy.bincount(work_r, self.data)
        counts = numpy.bincount(work_r)
        profile = intensities/counts

        val = reject_outliers(profile, m=5).std()
        print '{:10.3f} | {:10.0f}{:10.0f}{:10.6f}{:10.6f}'.format(val, cx, cy, rotx, roty)
        return val

    def start_coeffs(self, coeffs):
        return numpy.array(coeffs)/self.factors

    def final_coeffs(self, result):
        return self.factors * result.x

def radial(frame):
    # initialize
    cx, cy = 1350, 1180#frame.header['beam_center']
    nx, ny = frame.data.shape

    rotx, roty = 0.0, 0.0
    mask = (frame.data > 0.0) & (frame.data < frame.header['saturated_value'])
    work_data = frame.data[mask].ravel()

    # calculate radius
    rg = calc_r(nx, ny, cx, cy, rotx, roty).astype(numpy.int)
    org_r = rg[mask].ravel()

    r_max = org_r.max() + 1
    r_start = numpy.arange(r_max)

    # Calculate starting profile
    org_ints = numpy.bincount(org_r, work_data)
    org_counts = numpy.bincount(org_r)
    org_profile = org_ints/org_counts

    i = numpy.where(org_ints == org_ints.max())[0]
    r_bin = r_start[i]

    optimizer = StdOptimizer(work_data, nx, ny, mask, prec=numpy.array([1, 1, 0.1, 0.1]))
    coeffs = optimizer.start_coeffs([cx, cy, rotx, roty])
    results = optimize.minimize(
        optimizer, coeffs[:],
        args=(r_bin,),
        method='Nelder-Mead',
        # #method='Powell',
        # #method='L-BFGS-B',
        # #method='SLSQP',
        # bounds=bounds,
        options={
            'fatol': 0.5,
            'xtol': 0.01,
            #'ftol': 0.005,
            #'maxfev': 200,
            #'eps': 1e-5
        }
    )

    cx, cy, rotx, roty = optimizer.final_coeffs(results)

    print 'Final:    {:8.0f}{:8.0f}{:8.3f}{:8.3f}'.format(cx, cy, rotx, roty)

    r_opt = calc_r(nx, ny, cx, cy, rotx, roty).astype(numpy.int)
    work_r = r_opt[mask].ravel()
    r_max = work_r.max() + 1
    r = numpy.arange(r_max)

    # Calculate profile
    intensities = numpy.bincount(work_r, work_data)
    counts = numpy.bincount(work_r)
    return (r_start, org_profile), (r, intensities/counts)



if __name__ == '__main__':
    frame = imageio.read_image(sys.argv[1])
    (orx, ory), (rx, ry) = radial(frame)
    plt.plot(orx, ory, 'r')
    plt.plot(rx, ry, 'g')
    plt.show()
