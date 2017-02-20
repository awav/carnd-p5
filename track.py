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
import model
import os

import importlib

importlib.reload(model)
importlib.reload(common)

from calibrator import Calibrator as clb

class FrameVehiclePipeline():
    def __init__(self, classifier, shape=(720, 1280)):
        if classifier.__class__ is not model.CarModel:
           raise ValueError('You can pass only CarModel argument')
        self._model = classifier
        self._slices = Slices(**(self.slice_params()))
    def process(self, frame, show=False):
        im = clb.undistort(frame)
        boxes = self._find_cars_boxes(heatmap)
        return self._draw_car_boces(boxes)
    def slice_params(self, height=720, width=1280):
        ws = [64,96,128,160]
        nw_y = 400
        nw_xs = [300,200,100,0]
        se_ys = [nw_y + w for w in ws]
        boxes = [((nw_xs[0], nw_y), (width-nw_xs[0], se_ys[0])) for i in range(4)]
        #box1 = nw1, se1 = ((nw_xs[0], nw_y), (width-nw_xs[0], se_ys[0]))
        #box2 = nw2, se2 = ((nw_xs[1], nw_y), (width-nw_xs[1], se_ys[1]))
        #box3 = nw3, se3 = ((nw_xs[2], nw_y), (width-nw_xs[2], se_ys[2]))
        #box4 = nw4, se4 = ((nw_xs[3], nw_y), (width-nw_xs[3], se_ys[3]))
        #nw5, se5 = ((0, 400), (w, 592))
        return {'boxes': boxes,
                'windows': list(zip(ws, ws)),
                'overlaps': [(0.75, 0.75)] * 4}
    def _find_cars_boxes(im, show=False):
        heatmap = np.zeros(im.shape[:2])
        for nw, se in self._slices.wins:
            ys, ye = nw[1], se[1]
            xs, xe = nw[0], se[0]
            car = self._module.predict(np.resize(im[ys:ye,xs:xe,:], self._model.input_shape[:2]))
            if car:
                heapmap[ys:ye,xs:xe,:] += 1
        if show == True:
            common.show_images

class Slices():
    def __init__(self, **kwargs):
        self.wins = [w for w in self._gen_windows(**kwargs)]
    def _gen_windows(self,
                     boxes=[((0, 0),(1280,720))],
                     windows=[(64, 64)],
                     overlaps=[(0.5, 0.5)]):
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
