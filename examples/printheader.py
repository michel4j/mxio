#!/usr/bin/env python

import pprint
import sys

from .. import read_image

if __name__ == '__main__':
    frame = read_image(sys.argv[1])
    pprint.pprint(frame.header, indent=4)