import os
import re
import pytz
from datetime import datetime

import numpy


def stretch(gamma):
    lut = (gamma * numpy.arange(65536)).astype(numpy.uint)
    lut[lut > 254] = 254
    return lut


def calc_gamma(avg_int):
    return 2.7 if avg_int == 0.0 else 29.378 * avg_int ** -0.86


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
    p1 = re.compile('^(?P<root_name>[\w_-]+?)(?P<separator>(?:[._-])?)(?P<field>\d{3,12})(?P<extension>(?:\.[^\d][\w]+)?)$')

    m = p1.match(filename)
    if m:
        params = m.groupdict()
        files = os.listdir(directory)
        width = len(params['field'])
        current =  int(params['field'])
        regex = '^{root_name}{separator}(\d{{{width}}}){extension}$'.format(width=width, **params)
        p2 = re.compile(regex)
        frames = [int(m.group(1)) for f in files for m in [p2.match(f)] if m]

        template = '{root_name}{separator}{{field}}{extension}'.format(**params)
        name = template.format(field='{{:0{}d}}'.format(width))
        sequence = sorted(frames)
        first_file = name.format(sequence[0])
        return {
            'start_time': datetime.fromtimestamp(os.path.getmtime(os.path.join(directory, first_file)), tz=pytz.utc),
            'name': name,
            'label': params['root_name'],
            'directory': directory,
            'template': template.format(field='?'*width),
            'regex': regex,
            'reference': name.format(sequence[0]),
            'sequence': sequence,
            'current': current
        }
    return {}


def image_histogram(data):
    """
    Calculate and return the bins and edges histogram for the provided data
    """
    return numpy.histogram(data, density=False)
