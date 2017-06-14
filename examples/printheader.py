#!/usr/bin/env python

import imageio
import sys
import pprint

if __name__ == '__main__':
    frame = imageio.read_image(sys.argv[1])
    pprint.pprint(frame.header, indent=4)