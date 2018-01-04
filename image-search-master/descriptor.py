#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: ritesh
# @Date:   2015-08-21 14:34:44
# @Last Modified by:   ritesh
# @Last Modified time: 2015-08-21 14:39:14

# import the necessary packages
import numpy as np
import cv2

class RGBHistogram:
    def __init__(self, bins):
        """Store the number of bins the histogram will use"""
        self.bins = bins

    def describe(self, image):
        # compute a 3D histogram in the RGB colorspace,
        # then normalize the histogram so that images
        # with the same content, but either scaled larger
        # or smaller will have (roughly) the same histogram
        hist = cv2.calcHist([image], [0, 1, 2],
            None, self.bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist)

        # return out 3D histogram as a flattened array
        return hist.flatten()
