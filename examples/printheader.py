#!/usr/bin/env python

import pprint
import sys
import os
sys.path = [os.path.abspath('../..')] + sys.path
import imageio
from imageio import read_image
print imageio.__file__

if __name__ == '__main__':
    frame = read_image(sys.argv[1])
    pprint.pprint(frame.header, indent=4)