import os
import re
from typing import TypedDict
from datetime import datetime

import numpy
from numpy.typing import NDArray
import pytz


class ImageStats(TypedDict):
    maximum: float
    average: float
    minimum: float
    overloads: int


def image_stats(data: NDArray, saturated_value: float) -> ImageStats:
    """
    Calculate approximate image statistics using a quadrant of the image only

    :param data: NDArray
    :param saturated_value:
    :return: ImageStats dictionary
    """

    w, h = numpy.array(data.shape)//2
    stats_data = data[:h, :w]
    mask = stats_data > 0

    return {
        "maximum": stats_data[mask].max(),
        "average": stats_data[mask].mean(),
        "minimum": stats_data[mask].min(),
        "overloads": 4*(stats_data[mask] >= saturated_value).sum()
    }


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
    p1 = re.compile(
        r'^(?P<root_name>[\w_-]+?)(?P<separator>[._-]?)(?P<field>\d{3,12})(?P<extension>(?:\.\D\w+)?)$')

    m = p1.match(filename)
    if m:
        params = m.groupdict()
        files = os.listdir(directory)
        width = len(params['field'])
        current = int(params['field'])
        regex = r'^{root_name}{separator}(\d{{{width}}}){extension}$'.format(width=width, **params)
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
            'template': template.format(field='?' * width),
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
