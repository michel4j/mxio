import magic
import os
import re
from pathlib import Path

from .common import UnknownImageFormat, ImageIOError
from .formats import get_formats

MAGIC_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'magic')


def get_file_type(filename):
    try:
        m = magic.Magic(magic_file=MAGIC_FILE)
    except magic.MagicException:
        print('Magic file not found')
        m = magic.Magic()
    return m.from_file(filename).strip()


def read_image(path,  header_only=False):
    """
    Determine the file type using libmagic, open the image using the correct image IO
    back-end, and return an image object
    
    Every image Object has two attributes:
        - header: A dictionary 
        - image:  The actual PIL image of type 'I'
    """
    image_path = Path(path)

    # container formats have the index as a file within the image "directory"
    if re.match(r'^\d+$', image_path.name):
        filename = str(image_path.parent)
    else:
        filename = path

    formats = get_formats()
    full_id = get_file_type(filename)
    key = full_id.split(', ')[0].strip()
    obj_class = formats.get(key)

    if obj_class:
        img_obj = obj_class(path, header_only)
        return img_obj
    else:
        known_formats = ', '.join([v.split()[0] for v in formats.keys()])
        raise TypeError('Supported formats [{}]'.format(known_formats, ))


def read_header(path):
    img_obj = read_image(path, header_only=True)
    return img_obj.header


__all__ = ['UnknownImageFormat', 'ImageIOError', 'read_image', 'read_header']
