#!/usr/bin/env python

import functools
import operator
import sys
from imageio import read_image
import numpy
from matplotlib import pyplot as plt
from scipy import optimize, signal
import time
import os

XSTEP = 0.05
BACKSTOP_OFFSET = 100
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
    def __init__(self, frame, masks=[]):
        self.frame = frame
        self.nx, self.ny = frame.data.shape
        self.x_axis = numpy.arange(self.nx)
        self.y_axis = numpy.arange(self.ny)
        self.width = 10
        self.cx, self.cy = frame.header['beam_center']
        self.rotx = 1
        self.roty = 1
        self.masks = masks
        self.data_mask = (frame.data > 0.0) & (frame.data < frame.header['saturated_value'])
        self.pix_scale = frame.header['pixel_size'] / frame.header['distance']
        self.apply_masks()
        self.wedges = self.azimuth()
        print('Header beam center: {:0.0f}, {:0.0f}'.format(self.cx, self.cy))

    def apply_masks(self, gapfill=-2):
        for mask in self.masks:
            i0, j0, i1, j1 = mask
            self.frame.data[j0:j1, i0:i1] = gapfill

    def profile(self, sub_mask=None):
        mask = (self.frame.data > 0.0) & (self.frame.data < self.frame.header['saturated_value'])
        if sub_mask:
            mask &= sub_mask
        r = self.radii()[mask].ravel()
        data = self.frame.data[mask].ravel()
        prof = calc_profile(r, data)
        r_axis = numpy.arange(r.max() + 1)
        return numpy.array([r_axis, prof]).transpose()

    def azimuth(self):
        x = (self.x_axis - self.cx) / (0.5 * self.rotx * self.nx)
        y = (self.y_axis - self.cy) / (0.5 * self.roty * self.ny)
        return numpy.arctan2(x[:, None], y[None, :])

    def radii(self):
        x = (self.x_axis - self.cy) / self.rotx
        y = (self.y_axis - self.cx) / self.roty
        return numpy.hypot(x[:, None], y[None, :]).astype(numpy.int)

    def recenter(self, angle):
        a1 = angle % 360
        a2 = (angle + 180) % 360 - 360
        theta = self.wedges
        rg = self.radii()
        m1 = angle_slice(theta, a1, self.width) & self.data_mask
        m2 = angle_slice(theta, a2, self.width) & self.data_mask
        p1 = calc_profile(rg[m1], frame.data[m1].ravel())
        p2 = calc_profile(rg[m2], frame.data[m2].ravel())
        offset = calc_shift(p1[BACKSTOP_OFFSET:], p2[BACKSTOP_OFFSET:])
        dx = 0.5 * offset * numpy.cos(numpy.radians(a1))
        dy = 0.5 * offset * numpy.sin(numpy.radians(a1))
        self.cx, self.cy = map(int, (self.cx + dx, self.cy + dy))
        print('Optimized beam center: {}, {}'.format(self.cx, self.cy))

    def retilt(self, angle):
        a1 = 45
        a2 = 135
        theta = self.wedges
        rg = self.radii()
        m1 = angle_slice(theta, a1, self.width) & self.data_mask
        m2 = angle_slice(theta, a2, self.width) & self.data_mask
        p1 = calc_profile(rg[m1], frame.data[m1].ravel())
        p2 = calc_profile(rg[m2], frame.data[m2].ravel())
        offset = calc_shift(p1[BACKSTOP_OFFSET:], p2[BACKSTOP_OFFSET:])
        dx = 0.5 * offset * numpy.cos(numpy.radians(a1))
        dy = 0.5 * offset * numpy.sin(numpy.radians(a1))
        self.cx, self.cy = map(int, (self.cx + dx, self.cy + dy))
        print('Optimized beam center: {}, {}'.format(self.cx, self.cy))


def save_xdi(filename, profile):
    """Save  XDI 1.0 format"""
    fmt_prefix = '%10.3g'
    with open(filename, 'w') as f:
        f.write('# XDI/1.0 MXDC\n')
        f.write('# Mono.name: Si 111\n')

        for i, (info, units) in enumerate([('TwoTheta', 'degrees'), ('Intensity', 'counts')]):
            f.write('# Column.%d: %s %s\n' % (i + 1, info, units))

        f.write('#///\n')
        f.write('# %d data points\n' % len(profile))
        f.write('#---\n')
        numpy.savetxt(f, profile, fmt=fmt_prefix)

def radial_profile(frame):
    # initialize
    calc = FrameProfiler(frame, masks=MASK_REGIONS.get(frame.header['detector_type'], []))
    for angle in [45, 135, 45]:
        calc.recenter(angle)

    prof = calc.profile()
    prof[:, 0] = numpy.degrees(numpy.arctan(prof[:, 0] * calc.pix_scale))
    return prof


if __name__ == '__main__':
    import cProfile, pstats

    frame = read_image(sys.argv[1])
    # cProfile.run('prof = radial_profile(frame)', "{}.profile".format(__file__))
    # s = pstats.Stats("{}.profile".format(__file__))
    # s.strip_dirs()
    # s.sort_stats("time").print_stats(10)

    prof = radial_profile(frame)
    name = '{}.xdi'.format(os.path.splitext(os.path.basename(sys.argv[1]))[0])
    save_xdi(name, prof)

    #plt.plot(prof[:, 0], prof[:, 1])
    #plt.show()

