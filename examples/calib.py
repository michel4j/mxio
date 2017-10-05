#!/usr/bin/env python

import re
import sys
import os
sys.path = [os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))] + sys.path

import matplotlib
import numpy
from imageio import read_image

matplotlib.use('GTK3Agg')
from matplotlib import rcParams
from matplotlib import pyplot as plt
from scipy import signal, interp, optimize

rcParams['legend.loc'] = 'best'
rcParams['figure.facecolor'] = 'white'
rcParams['figure.edgecolor'] = 'white'
rcParams['font.sans-serif'] = 'Cantarell'
rcParams['font.size'] = 12
rcParams['figure.figsize'] = [16.0, 8.0]

XSTEP = 0.05
BACKSTOP_OFFSET = 200
CENTER_SEARCH_LIMIT = 500
MASK_REGIONS = {
    'pilatus36m': [(423, 2400, 523, 2462)],
}


def peak_width(a, x, w=30):
    left = numpy.where(a[x - w:x] > a[x] / 2)
    right = numpy.where(a[x:x + w] > a[x] / 2)
    left = w if left else left[0][0]
    right = 0 if not right else right[0][-1]
    return right + w - left


def find_peaks(y, width=9, sensitivity=0.01, smooth=True):
    ypp = signal.savgol_filter(y, width, 2, deriv=2)
    ypp[ypp > 0] = 0.0
    ypp *= -1
    yp = signal.savgol_filter(ypp, width, 1, deriv=1)
    hw = width // 2
    peak_patt = "(H{%d}.L{%d})" % (hw - 1, hw - 1)
    ps = ""
    for v in yp:
        if v == 0.0:
            ps += '-'
        elif v > 0.0:
            ps += 'H'
        else:
            ps += 'L'

    def get_peak(pos):
        return pos, y[pos], ypp[pos]

    peak_positions = [get_peak(m.start() + hw - 1) for m in re.finditer(peak_patt, ps)]
    ymax = max(y)
    yppmax = max(ypp)
    return numpy.array([
        v[:2] for v in peak_positions if (v[1] >= sensitivity * ymax and v[2] > 0.5 * sensitivity * yppmax)
    ])


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
    # ns = numpy.copy(s[BACKSTOP_OFFSET:])
    ns = numpy.log(1 + 1000 * (s - s.min()) / (s.max() - s.min()))
    return ns


def calc_shift(original, match):
    s1 = norm_curve(original)
    s2 = norm_curve(match)
    sel = numpy.isfinite(s1) & numpy.isfinite(s2)
    z = signal.fftconvolve(s1[sel], s2[sel][::-1])
    lags = numpy.arange(z.size) - (s2.size - 1)
    return (lags[numpy.argmax(numpy.abs(z))])


def calc_scale(original, match):
    s1 = norm_curve(original)
    s2 = norm_curve(match)
    sel = numpy.isfinite(s1) & numpy.isfinite(s2)
    params = [0, 1.0]
    x = numpy.arange(sel.sum())
    out, result = optimize.leastsq(prof_err, params[:], args=(match, original, x), maxfev=25000)
    offset, scale = out
    return offset, scale


def corr_shift(s1, s2):
    corr = signal.correlate(s1, s2)
    return numpy.where(corr == corr.max())[0] % s1.shape[0]


def prof_err(coeffs, p1, p2, x):
    offset, scale = coeffs[:]
    ynew = interp((x + offset) * scale, p1, x)
    return (numpy.log(p2) - numpy.log(ynew)) * p1


class PeakSorter(object):
    def __init__(self):
        self.tree = None
        self.distances = None

    def add_peaks(self, peaks):
        if not self.tree:
            self.tree = [
                [pk] for pk in peaks
            ]
            self.distances = [
                [0] for pk in peaks
            ]
        else:
            found = []
            for peak in peaks:
                idx, min_d = self.closest(peak)
                if idx >= 0 and idx not in found:
                    found.append(idx)
                    self.tree[idx].append(peak)
                    self.distances[idx].append(min_d)

    def closest(self, pk):
        ds = [
            numpy.hypot(*(pk - pks[-1]))
            for pks in self.tree
        ]
        min_d = min(ds)
        return ds.index(min_d), min_d


class FrameProfiler(object):
    def __init__(self, filename):
        self.frame = read_image(filename)
        self.masks = MASK_REGIONS.get(self.frame.header['detector_type'], [])
        self.nx, self.ny = self.frame.data.shape
        self.x_axis = numpy.arange(self.nx)
        self.y_axis = numpy.arange(self.ny)
        self.cx, self.cy = self.frame.header['beam_center']
        self.rho_x, self.rho_y = 0.0, 0.0
        self.apply_masks()
        self.mask = (self.frame.data > 0.0) & (self.frame.data < self.frame.header['saturated_value'])

    def apply_masks(self, gapfill=-2):
        for mask in self.masks:
            i0, j0, i1, j1 = mask
            self.frame.data[j0:j1, i0:i1] = gapfill

    def point_transform(self, size=180, rhos=(None, None)):
        rsize, asize = max(self.nx, self.ny) / 2, size

        r = self.radii(rho_x=rhos[0], rho_y=rhos[1])[self.mask]
        a = self.azimuth(rho_x=rhos[0], rho_y=rhos[1])[self.mask]

        rmin, rmax = r.min(), r.max()
        amin, amax = a.min(), a.max()

        self.ri = numpy.linspace(rmin, rmax, rsize)
        self.ai = numpy.linspace(amin, amax, asize)

        self.PX, self.PY = self.cart(self.ri, self.ai)
        self.rdata = numpy.zeros((asize, rsize))
        for i in range(asize):
            for j in range(rsize):
                if self.PX[i, j] < 0 or self.PY[i, j] < 0: continue
                if self.PX[i, j] >= self.nx or self.PY[i, j] >= self.ny: continue
                self.rdata[i, j] = self.frame.data[int(self.PX[i, j]), int(self.PY[i, j])]

    def transform(self, size=180, rhos=(None, None)):
        rsize, asize = max(self.nx, self.ny), size

        r = self.radii(rho_x=rhos[0], rho_y=rhos[1])[self.mask]
        a = self.azimuth(rho_x=rhos[0], rho_y=rhos[1])[self.mask]

        rmin, rmax = r.min(), r.max()
        amin, amax = a.min(), a.max()

        self.ri = numpy.linspace(rmin, rmax, rsize)
        self.ai = numpy.linspace(amin, amax, asize)

        self.PX, self.PY = self.cart(self.ri, self.ai)
        self.rdata = numpy.zeros((asize, rsize))
        for i in range(asize):
            for j in range(rsize):
                if self.PX[i, j] < 0 or self.PY[i, j] < 0: continue
                if self.PX[i, j] >= self.nx or self.PY[i, j] >= self.ny: continue
                self.rdata[i, j] = self.frame.data[int(self.PX[i, j]), int(self.PY[i, j])]

    def shiftr(self, data):
        asize, rsize = data.shape
        offsets = numpy.zeros((asize,))
        last_i = 0
        sel = numpy.isfinite(data[0, :])
        last_avg = data[0, sel].mean()
        while last_avg < 0:
            last_i += 1
            sel = numpy.isfinite(data[0, :])
            last_avg = data[last_i, sel].mean()

        for i in range(last_i + 1, asize):
            sel = numpy.isfinite(data[0, :])
            avg = data[i, sel].mean()
            if (avg / last_avg) < 0.5 or last_avg < 0:
                continue
            else:
                offset = calc_shift(data[last_i, :], data[i, :])
                if offset < -20 or offset > 20: continue
                offsets[i] = offset
                last_i = i
                last_avg = avg

        offsets = numpy.cumsum(offsets).astype(int)
        pad = abs(min(0, offsets.min()))
        offsets += pad
        sdata = numpy.zeros_like(data)
        for i, offset in enumerate(offsets):
            sdata[i, offset:] = data[i, :(rsize - offset)]
        return sdata, offsets

    def scaler(self, data):
        asize, rsize = data.shape
        offsets = numpy.zeros((asize,))
        scales = numpy.zeros((asize,))
        last_i = 0
        sel = numpy.isfinite(data[0, :])
        last_avg = data[0, sel].mean()
        while last_avg < 0:
            last_i += 1
            sel = numpy.isfinite(data[0, :])
            last_avg = data[last_i, sel].mean()

        for i in range(last_i + 1, asize):
            sel = numpy.isfinite(data[0, :])
            avg = data[i, sel].mean()
            if (avg / last_avg) < 0.5 or last_avg < 0:
                continue
            else:
                offset, scale = calc_scale(data[last_i, :], data[i, :])
                if offset < -20 or offset > 20: continue
                offsets[i] = offset
                scales[i] = scale
                last_i = i
                last_avg = avg

        offsets = numpy.cumsum(offsets).astype(int)
        pad = abs(min(0, offsets.min()))
        offsets += pad
        sdata = numpy.zeros_like(data)
        for i, offset in enumerate(offsets):
            sdata[i, offset:] = data[i, :(rsize - offset)]
        return sdata, offsets

    def profile(self, sub_mask=None):
        mask = (self.frame.data > 0.0) & (self.frame.data < self.frame.header['saturated_value'])
        if sub_mask is not None:
            mask &= sub_mask
        r = self.radii()[mask].ravel()
        data = self.frame.data[mask].ravel()
        prof = calc_profile(r.astype(int), data)
        r_axis = numpy.arange(r.max())
        fprof = numpy.dstack([r_axis, prof])[0]
        return fprof

    def from_polar(self, r, theta):
        return self.cx + r * numpy.cos(theta), self.cy + r * numpy.sin(theta)

    def cart(self, r, theta):
        return self.cx + r[None, :] * numpy.cos(theta[:, None]), self.cy + r[None, :] * numpy.sin(theta[:, None])

    def azimuth(self, rho_x=None, rho_y=None):
        if rho_x is None or rho_y is None:
            rho_x, rho_y = self.rho_x, self.rho_y
        x = (self.x_axis - self.cx) * numpy.cos(rho_x)
        y = (self.y_axis - self.cy) * numpy.cos(rho_y)
        return numpy.arctan2(x[:, None], y[None, :])

    def radii(self, rho_x=None, rho_y=None):
        if rho_x is None or rho_y is None:
            rho_x, rho_y = self.rho_x, self.rho_y
        x = (self.x_axis - self.cx) * numpy.cos(rho_x)
        y = (self.y_axis - self.cy) * numpy.cos(rho_y)
        return (numpy.hypot(x[:, None], y[None, :]))

    def quality(self, rhos):
        self.transform(size=30, rhos=(rhos[0], rhos[0]))
        data = self.rdata
        sel = data > 0
        data[data <= 0] = 0
        prof = data.sum(axis=0) / sel.sum(axis=0)
        spks = numpy.array(find_peaks(prof))
        score = numpy.array([peak_width(prof, x) + 1 for x in spks[:, 0].astype(int)])
        return 1.0 / score


def radial_profile(filename):
    # initialize
    calc = FrameProfiler(filename)

    # pass 1
    cx = calc.cx
    cy = calc.cy
    calc.point_transform(size=30)
    orig_data = calc.rdata

    for i in range(2):
        calc.cx = cx
        calc.cy = cy
        calc.point_transform(size=30)
        data = calc.rdata
        data, offsets = calc.shiftr(data)

        adx, ady = calc.from_polar(-calc.ri[offsets], calc.ai)
        vx, vy = calc.cx - adx.mean(), calc.cy - ady.mean()
        cx = calc.cx - 2 * vx
        cy = calc.cy - 2 * vy
        print 'Adjusted center', cx, cy

    calc.cx = cx
    calc.cy = cy
    calc.point_transform(size=90, rhos=(numpy.pi / 360, numpy.pi / 360))
    data = calc.rdata

    # circles
    sel = data > 0
    data[data <= 0] = 0
    prof = data.sum(axis=0) / sel.sum(axis=0)
    spks = find_peaks(prof)

    # image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(numpy.log(calc.frame.data.T), origin='upper')
    ax.plot([calc.cx, cx], [calc.cy, cy], '+')
    for x in spks[:,0].astype(int):
        r = calc.ri[x]
        c = plt.Circle((calc.cx, calc.cy), r, fill=False, color='#ff0000', lw=0.5, alpha=0.5)
        ax.add_artist(c)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.imshow(numpy.log(orig_data), aspect='auto')
    ax2 = fig.add_subplot(312)
    ax2.imshow(numpy.log(data), aspect='auto')
    ax3 = fig.add_subplot(313)
    for x in spks[:, 0].astype(int):
        if x < BACKSTOP_OFFSET: continue
        w = peak_width(prof, x) + 1
        ax3.axvline(x, lw=w + 1, color='#ff0000', alpha=0.5)

    ax3.set_xlim(0, data.shape[1])
    ax3.plot(prof)
    plt.show()

    # for rx in numpy.linspace(-numpy.pi/100, numpy.pi/100, 5):
    #     print numpy.radians((rx, 0))
    #     calc.point_transform(size=90, rhos=(rx, rx))
    #     data = calc.rdata
    #     sel = data > 0
    #     data[data<=0] = 0
    #     prof = data.sum(axis=0) / sel.sum(axis=0)
    #     spks = numpy.array(find_peaks(prof))
    #     widths = numpy.array([peak_width(prof, x) + 1 for x in spks[:, 0].astype(int)]).sum()
    #     print spks.shape[0], widths
    #     fig = plt.figure()
    #     ax = fig.add_subplot(211)
    #     ax.imshow(numpy.log(odata), aspect='auto')
    #     ax2 = fig.add_subplot(212)
    #     ax2.imshow(numpy.log(data), aspect='auto')
    #     plt.show()

    prof = calc.profile()
    plt.plot(prof[:, 0], prof[:, 1])
    plt.show()


if __name__ == '__main__':
    # import cProfile, pstats
    # cProfile.run('prof = radial_profile(sys.argv[1])', "{}.profile".format(__file__))
    # s = pstats.Stats("{}.profile".format(__file__))
    # s.strip_dirs()
    # s.sort_stats("time").print_stats(10)

    radial_profile(sys.argv[1])
