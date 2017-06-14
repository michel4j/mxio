#!/usr/bin/env python

import functools
import operator
import sys

from .. import read_image
import numpy
from matplotlib import pyplot as plt
from scipy import optimize, signal

XSTEP = 0.05


def baseline(values):
    cdev = values.std()
    orig = len(values)
    pdev = cdev + 100
    mx = values.max()
    delta = 100.
    while (abs(pdev - cdev) > 0.1) and (mx > values.min() + cdev*2):
        pdev = cdev
        values = values[values<mx]
        mx = values.max()
        cdev = values.std()
    return cdev


def slices(x, n, m):
    slice = 2*numpy.pi/m
    ranges = [
        (start_angle, start_angle + slice)
        for start_angle in numpy.linspace(-numpy.pi, numpy.pi, m+2)[1:]
    ]
    return [
        (x > start_angle) & (x < end_angle)
        for start_angle, end_angle in ranges
    ]

def select_slices(x, n, m):
    slice = 2*numpy.pi/m
    ranges = [
        (start_angle, start_angle + slice)
        for start_angle in numpy.linspace(-numpy.pi, numpy.pi, m + 2)[1:]
    ]
    return functools.reduce(operator.__or__, [
        (x > start_angle) & (x < end_angle)
        for start_angle, end_angle in ranges
    ])


def reject_outliers(data, m=2.):
    d = numpy.abs(data - numpy.median(data))
    mdev = numpy.median(d)
    s = d / mdev if mdev else 0.
    return data[s < m]


def calc_r(nx, ny, cx, cy, rotx=0, roty=0):
    x = (numpy.arange(nx) - int(cx)) / numpy.cos(rotx)
    y = (numpy.arange(ny) - int(cy)) / numpy.cos(roty)
    return numpy.hypot(x[:, None], y[None, :])


def calc_theta(nx, ny, cx, cy, rotx=0, roty=0):
    x = (numpy.arange(nx) - int(cx)) / numpy.cos(rotx)
    y = (numpy.arange(ny) - int(cy)) / numpy.cos(roty)
    return numpy.arctan2(2 * x[:, None] / nx, 2 * y[None, :] / ny)


def calc_profile(r, data):
    intensities = numpy.bincount(r, data)
    counts = numpy.bincount(r)
    return intensities / counts

class StdOptimizer(object):
    def __init__(self, data, nx, ny, mask, prec=1):
        self.mask = mask
        self.data = data
        self.nx, self.ny = nx, ny
        self.factors = 1.0 / (prec * XSTEP)

    def __call__(self, coeffs):
        pars = self.factors * coeffs

        rg = calc_r(self.nx, self.ny, *pars).astype(numpy.int)
        work_r = rg[self.mask].ravel()
        profile = calc_profile(work_r, self.data)
        val = len(signal.find_peaks_cwt(profile, numpy.arange(1, 5)))
        print '{:10.3f} | {:10.0f}{:10.0f}'.format(val, *pars)
        return val

    def start_coeffs(self, coeffs):
        return numpy.array(coeffs) / self.factors

    def final_coeffs(self, result):
        return self.factors * result.x


def radial(frame):
    # initialize
    cx, cy = frame.header['beam_center']
    nx, ny = frame.data.shape
    rotx, roty = 0.0, 0.0

    theta = calc_theta(nx, ny, cx, cy)
    theta_mask = select_slices(theta, 9, 8)
    mask = (frame.data > 0.0) & (frame.data < frame.header['saturated_value'])
    mask &= theta_mask

    # Calculate starting profile
    rg = calc_r(nx, ny, cx, cy).astype(numpy.int)
    org_r = rg[mask].ravel()
    r_max = org_r.max() + 1
    r_start = numpy.arange(r_max)
    work_data = frame.data[mask].ravel()
    org_profile = calc_profile(org_r, work_data)

    # Optimize parameters
    optimizer = StdOptimizer(work_data, nx, ny, mask, prec=numpy.array([1, 1]))
    coeffs = optimizer.start_coeffs([cx, cy])
    results = optimize.minimize(
        optimizer, coeffs[:],
        method='Nelder-Mead',
        #method='Powell',
        # #method='L-BFGS-B',
        #method='SLSQP',
        # bounds=bounds,
        options={
            'fatol': 0.5,
            'xtol': .02,
            # 'ftol': 0.005,
            # 'maxfev': 200,
            # 'eps': 1e-5
        }
    )

    pars = optimizer.final_coeffs(results)
    print 'Final:    {:8.0f}{:8.0f}'.format(*pars)
    plots = [(r_start, org_profile)]
    r_opt = calc_r(nx, ny, *pars).astype(numpy.int)
    work_r = r_opt[mask].ravel()
    r = numpy.arange(work_r.max() + 1)
    plots.append((r, calc_profile(work_r, work_data)))

    # for slice in slices(theta, 18, 4):
    #     this_mask = mask & slice
    #     if not this_mask.sum(): continue
    #     work_data = frame.data[this_mask].ravel()
    #     work_r = r_opt[this_mask].ravel()
    #     r = numpy.arange(work_r.max() + 1)
    #     plots.append((r, calc_profile(work_r, work_data)))


    return plots

if __name__ == '__main__':
    frame = read_image(sys.argv[1])
    plots = radial(frame)
    for orx, ory in plots:
        print '---------------'
        plt.plot(orx, ory)
        baseline(ory)

    plt.show()
