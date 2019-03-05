import os
import re

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
    xnew, ynew = numpy.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    tck = interpolate.bisplrep(x, y, z, s=0)
    znew = interpolate.bisplev(xnew[:, 0], ynew[0, :], tck)
    return znew


def file_sequences(filename):
    """
    Determine the parameters of a full sequence of files for which a single file belongs. For example, passing in the
    file "data_00002.h5", will produce the dictionary     {'name': "data_{:05d}.h5", 'template': "data_?????.h5",
    "sequence": [1,2,3,4], "current": 2} if the directory also contains files "data_00002.h5", "data_00003.h5",
    "data_00004.h5".

    :param filename:  full path of one file belonging to the series
    :return: dictionary describing the sequence or an empty dictionary if file does not belong to a sequence
    """

    directory, filename = os.path.split(os.path.abspath(filename))
    p1 = re.compile('^(?P<root_name>.+)_(?P<number>\d{3,7})\.?(?P<extension>[\w]+)?$')
    m = p1.match(filename)
    if m:
        params = m.groupdict()
        files = os.listdir(directory)
        width = len(params['number'])
        current =  int(params['number'])
        p2 = re.compile('{root_name}_(?P<number>\d{{{width}}}).{extension}'.format(width=width, **params))
        print p2.pattern
        frames = [int(m.group(1)) for f in files for m in [p2.match(f)] if m]

        template = os.path.join(directory, '{root_name}_{{field}}.{extension}'.format(**params))
        return {
            'name': template.format(field='{{:0{}d}}'.format(width)),
            'template': template.format(field='?'*width),
            'sequence': sorted(frames),
            'current': current
        }
    return {}
