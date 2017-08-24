#!/usr/bin/env python

import functools
import operator
import sys
from imageio import read_image
import numpy
import matplotlib

matplotlib.use('GTK3Agg')
from matplotlib import colors, gridspec, rcParams
from matplotlib import pyplot as plt
from scipy import optimize, signal, interp
import os
from multiprocessing import cpu_count, Pool, Value
from multiprocessing.managers import BaseManager

rcParams['legend.loc'] = 'best'
rcParams['legend.isaxes'] = False
rcParams['figure.facecolor'] = 'white'
rcParams['figure.edgecolor'] = 'white'
rcParams['font.sans-serif'] = 'Cantarell'
rcParams['font.size'] = 12
rcParams['figure.figsize'] = [16.0, 8.0]

XSTEP = 0.05
BACKSTOP_OFFSET = 100
CENTER_SEARCH_LIMIT = 500
MASK_REGIONS = {
    'pilatus36m': [(2400, 423, 2462, 523)],
}


def baseline(values):
    cdev = values.std()
    orig = len(values)
    pdev = cdev + 100
    mx = values.max()
    delta = 3
    while (abs(pdev - cdev) > 0.1) and (mx > values.min() + cdev * 2):
        pdev = cdev
        values = values[values < mx]
        mx = values.max()
        cdev = values.std()
    return values


def angle_slice(x, angle, width=1):
    angle = angle if angle <= 180 else angle - 360.0
    slice = numpy.radians(angle)
    w = numpy.radians(width) / 2
    return (x > slice - w) & (x < slice + w)


def calc_profile(r, data):
    intensities = numpy.bincount(r, data)
    counts = numpy.bincount(r) + 1
    return (intensities / counts)


def norm_curve(s):
    return (s - s.min()) / (s.max() - s.min())


def calc_shift(original, match):
    s1 = norm_curve(original)
    s2 = norm_curve(match)
    z = signal.fftconvolve(s1, s2[::-1])
    lags = numpy.arange(z.size) - (s2.size - 1)
    return (lags[numpy.argmax(numpy.abs(z))])


class FrameProfiler(object):
    def __init__(self, filename):
        self.frame = read_image(filename)
        self.masks = MASK_REGIONS.get(self.frame.header['detector_type'], [])
        self.nx, self.ny = self.frame.data.shape
        self.x_axis = numpy.arange(self.nx)
        self.y_axis = numpy.arange(self.ny)
        self.width = 15
        self.cx, self.cy = self.frame.header['beam_center']
        self.rotx = 0.0
        self.roty = 0.0
        self.xskew = 1.0
        self.yskew = 1.0

        self.data_mask = (self.frame.data > 0.0) & (self.frame.data < self.frame.header['saturated_value'])
        self.pix_scale = self.frame.header['pixel_size'] / self.frame.header['distance']
        self.apply_masks()
        self.wedges = self.azimuth()
        self.scales = numpy.ones_like(self.wedges)
        print('Header beam center: {:0.0f}, {:0.0f}'.format(self.cx, self.cy))

    def apply_masks(self, gapfill=-2):
        for mask in self.masks:
            i0, j0, i1, j1 = mask
            self.frame.data[j0:j1, i0:i1] = gapfill

    def profile(self, sub_mask=None):
        mask = (self.frame.data > 0.0) & (self.frame.data < self.frame.header['saturated_value'])
        if sub_mask is not None:
            mask &= sub_mask
        r = self.radii()[mask].ravel()
        data = self.frame.data[mask].ravel()
        prof = calc_profile(r, data)
        prof[:BACKSTOP_OFFSET] = prof[BACKSTOP_OFFSET]

        r_axis = numpy.arange(r.max() + 1)
        return numpy.array([r_axis, prof]).transpose()

    def from_polar(self, r, theta):
        return self.cx + r * numpy.cos(theta), self.cy + r * numpy.sin(theta)

    def azimuth(self):
        x = (self.x_axis - self.cx) / (0.5 * self.nx)
        y = (self.y_axis - self.cy) / (0.5 * self.ny)
        xm = self.xskew ** numpy.sign(x)
        ym = self.yskew ** numpy.sign(y)
        x = (x * xm)
        y = (y * ym)
        return numpy.arctan2(x[:, None], y[None, :])

    def radii(self):
        x = (self.x_axis - self.cy)
        y = (self.y_axis - self.cx)
        xm = self.xskew ** numpy.sign(x)
        ym = self.yskew ** numpy.sign(y)
        x = (x * xm)
        y = (y * ym)
        return (numpy.hypot(x[:, None], y[None, :])).astype(numpy.int)

    def recenter(self, angle):
        a1 = angle % 360
        a2 = (angle + 180) % 360 - 360
        theta = self.azimuth()
        rg = self.radii()
        rmask = rg < rg.max() / 2
        m1 = angle_slice(theta, a1, self.width) & self.data_mask & rmask
        m2 = angle_slice(theta, a2, self.width) & self.data_mask & rmask
        p1 = calc_profile(rg[m1], self.frame.data[m1].ravel())
        p2 = calc_profile(rg[m2], self.frame.data[m2].ravel())
        offset = calc_shift(
            p1[BACKSTOP_OFFSET:CENTER_SEARCH_LIMIT + BACKSTOP_OFFSET],
            p2[BACKSTOP_OFFSET:CENTER_SEARCH_LIMIT + BACKSTOP_OFFSET]
        )

        dx = 0.5 * offset * numpy.cos(numpy.radians(a1))
        dy = 0.5 * offset * numpy.sin(numpy.radians(a1))
        self.cx, self.cy = map(int, (self.cx + dx, self.cy + dy))
        print('Optimized beam center: {}, {}'.format(self.cx, self.cy))

    def rescale(self, a1, a2):
        theta = self.azimuth()
        m1 = angle_slice(theta, a1, self.width)
        m2 = angle_slice(theta, a2, self.width)
        p1 = self.profile(m1)
        p2 = self.profile(m2)
        params = [1.00, 0]
        out, result = optimize.leastsq(prof_err, params[:], args=(p1, p2), maxfev=25000)
        return out

    def retilt(self):
        scale, offset = self.rescale(45, -135)
        self.cx += offset * numpy.sin(numpy.pi / 4)
        self.xskew = scale**0.25
        scale, offset = self.rescale(90, -90)
        self.cy += offset / 2
        self.yskew = scale**0.5
        self.roty = numpy.arctan(self.yskew) - numpy.pi / 4
        self.rotx = numpy.arctan(self.xskew) - numpy.pi / 4
        print(
            'Optimized Parameters: cx={:0.1f}, cy={:0.1f}, rotx={:0.4f} deg, roty={:0.4f} deg'.format(
                self.cx, self.cy, numpy.degrees(self.rotx), numpy.degrees(self.roty)
            )
        )


def save_xdi(filename, profile):
    """Save  XDI 1.0 format"""
    fmt_prefix = '%10.6f'
    with open(filename, 'w') as f:
        f.write('# XDI/1.0 MXDC\n')
        f.write('# Mono.name: Si 111\n')

        for i, (info, units) in enumerate([('TwoTheta', 'degrees'), ('Intensity', 'counts')]):
            f.write('# Column.%d: %s %s\n' % (i + 1, info, units))

        f.write('#///\n')
        f.write('# %d data points\n' % len(profile))
        f.write('#---\n')
        numpy.savetxt(f, profile, fmt=fmt_prefix)


def interp_prof(p1, p2, coeffs):
    scale, offset = coeffs[:]
    p3 = numpy.zeros_like(p1)
    p3[:, 0] = p1[:, 0]
    p3[:, 1] = interp(p1[:, 0], (p2[:, 0] + offset) * scale, p2[:, 1])
    return p3


def prof_err(coeffs, p1, p2):
    scale, offset = coeffs[:]
    ynew = interp(p1[:, 0], (p2[:, 0] + offset) * scale, p2[:, 1])
    return (numpy.log(p1[:, 1]) - numpy.log(ynew)) * numpy.log(p1[:, 1])


def radial_profile(filename):
    # initialize
    calc = FrameProfiler(filename)
    calc.retilt()
    # for angle in [45, 135, 45, 135]:
    #    calc.recenter(angle)

    # swidth = 15
    # angles = [(30, a) for a in numpy.arange(30, 330, swidth)]
    # results = []
    # for a1, a2 in angles:
    #     scale, offset = calc.rescale(a1, a2)
    #     res = [a1, a2, offset, scale]
    #     m = angle_slice(calc.wedges, a2)
    #     calc.scales[m] = scale
    #     results.append(res)
    #     print res
    # numpy.savetxt('junk.dat', results)
    # prof = calc.profile(angle_slice(calc.wedges, angle, swidth))
    # plt.plot((prof[:, 0] + offset)*scale, prof[:, 1] + 20.0 * i)
    # print offset, science.find_peaks(prof[:, 0], prof[:, 1])
    #plt.show()
    prof = calc.profile()
    prof[:, 0] = numpy.degrees(numpy.arctan(prof[:, 0] * calc.pix_scale))
    return prof


if __name__ == '__main__':
    #import cProfile, pstats
    # cProfile.run('prof = radial_profile(sys.argv[1])', "{}.profile".format(__file__))
    # s = pstats.Stats("{}.profile".format(__file__))
    # s.strip_dirs()
    # s.sort_stats("time").print_stats(10)

    prof = radial_profile(sys.argv[1])
    name = '{}.xdi'.format(os.path.splitext(os.path.basename(sys.argv[1]))[0])
    save_xdi(name, prof)

    plt.plot(prof[:, 0], prof[:, 1], 'c-')
    plt.show()
