# The MIT License (MIT)
#
# Copyright 2017 Artem Artemev, im@artemav.com
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import common
import features
import model as mdl
import os

import importlib

importlib.reload(model)
importlib.reload(common)

from calibrator import Calibrator as clb

class FrameVehiclePipeline():
    def __init__(self, model, shape=(720, 1280)):
        self.
        if model.__class__ is not mdl.CarModel:
           raise ValueError('You can pass only CarModel argument')
        self._model = model
    def process(self, frame, show=False):
        im = clb.undistort(frame)
    def windows(img):
        
    def _windows(boxes=[((0, 0),(720,1280))], windows=[(64, 64)], overlaps=[(0.5, 0.5)]):
        n = len(boxes)
        assert(n == len(windows) and n == len(overlaps))
        for i in range(n):
            top = boxes[i][0]
            bot = boxes[i][1]
            window = windows[i]
            overlap = overlaps[i]
            ##
            height = bot[0] - top[0]
            width = bot[1] - top[1]
            xstep = np.int(window[0]*(1-overlap[0]))
            ystep = np.int(window[1]*(1-overlap[1]))
            xwins = np.int(width/xstep) - 1
            ywins = np.int(height/ystep) - 1
            window_list = []
            for x in range(xwins):
                for y in range(ywins):
                    x_beg = x * xstep + top[0]
                    x_end = x_beg + window[0]
                    y_beg = y * ystep + top[1]
                    y_end = y_beg + window[1]
                    yield ((x_beg, y_beg), (x_end, y_end))
