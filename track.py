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
import scipy.ndimage.measurements as measurments

import importlib

importlib.reload(model)
importlib.reload(common)

from calibrator import Calibrator as clb

class FrameVehiclePipeline():
    def __init__(self, classifier, shape=(720, 1280)):
        if classifier.__class__ is not model.CarModel:
           raise ValueError('You can pass only CarModel argument')
        self._model = classifier
        self._slicer = Slicer(**(self.slice_params()))
        self._heatmap = np.zeros(shape)
        self._labels = None
        self._i = 0
    def process(self, orig, show=False):
        im = clb.undistort(orig, show=show)
        self._find_cars_heatmap(im, show=show)
        self._find_cars_boxes(show=show)
        self._reset_heatmap()
        return self._draw_car_boxes(orig, show=show)
        #return self._heatmap
    def slice_params(self, height=720, width=1280):
        n = 4
        ws = [96,128,160,192]
        #nw_y = [400,400,430,460]
        nw_y = [376,368,390,412]
        nw_xs = [0,0,0,0]
        se_ys = [500,500,550,580]
        #n = 3
        #ws = [96,128,160]
        #nw_y = [400,400,400]
        #nw_xs = [100,0,0]
        #se_ys = [nw_y[i] + ws[i] for i in range(n)]

        boxes = [((nw_xs[i], nw_y[i]), (width-nw_xs[i], se_ys[i])) for i in range(n)]
        return {'boxes': boxes,
                'windows': list(zip(ws, ws)),
                'overlaps': [(0.75, 0.75)] * n}
    def samples(self, im, target='data/video_dataset/'):
        im = clb.undistort(im)
        shape = self._model.input_shape
        for nw, se in self._slicer.wins:
            ys, ye = nw[1], se[1]
            xs, xe = nw[0], se[0]
            cpy = cv.resize(im[ys:ye,xs:xe,:], shape[:2])
            cpy = common.cvt_color(cpy, color='BGR', src='RGB')
            cv.imwrite('{0}/{1}.png'.format(target, self._i), cpy)
            self._i += 1
        print(self._i)
        return im
    def _find_cars_heatmap(self, im, show=False):
        shape = self._model.input_shape
        #_show = show if show == True else False
        _show = False
        for nw, se in self._slicer.wins:
            ys, ye = nw[1], se[1]
            xs, xe = nw[0], se[0]
            #print(nw, se)
            car = self._model.predict(cv.resize(im[ys:ye,xs:xe,:], shape[:2]), show=_show)
            #_show = False
            if car == 1:
                self._heatmap[ys:ye,xs:xe] += 1
                if show == True:
                    cv.rectangle(im, nw, se, (0,0,255), 2)
                    #common.show_image(im[ys:ye,xs:xe,:], titles='resized-car')
        if show == True:
            common.show_image([im, self._heatmap], ncols=2, window_title='Cars Heat Map',
                              titles=['original', 'heatmap'])
    def _find_cars_boxes(self, thresh=9, show=False):
        self._heatmap[self._heatmap < thresh] = 0
        self._labels = measurments.label(self._heatmap)
    def _draw_car_boxes(self, im, show=False):
        n = self._labels[1]+1
        for car in range(1, n):
            ## x and y coordinates
            y, x = (self._labels[0] == car).nonzero()
            nw, se = (np.min(x), np.min(y)), (np.max(x), np.max(y))
            cv.rectangle(im, nw, se, (255,255,0), 2)
        if show == True:
            common.show_image(im, window_title='Cars Heat Map', titles='Detected cars')
        return im
    def _reset_heatmap(self):
        self._heatmap[:] *= 0.5

class Slicer():
    def __init__(self, **kwargs):
        self.wins = [w for w in self._gen_windows(**kwargs)]
    def _gen_windows(self,
                     boxes=[((0, 0),(1280,720))],
                     windows=[(64, 64)],
                     overlaps=[(0.5, 0.5)]):
        n = len(boxes)
        assert(n == len(windows) and n == len(overlaps))
        for i in range(n):
            nw, se = boxes[i]
            window = windows[i]
            overlap = overlaps[i]
            ##
            width = se[0] - nw[0]
            height = se[1] - nw[1]
            xstep = np.int(window[0]*(1-overlap[0]))
            ystep = np.int(window[1]*(1-overlap[1]))
            xwins = np.int(width/xstep) - 1
            ywins = np.int(height/ystep) - 1
            window_list = []
            for x in range(xwins):
                for y in range(ywins):
                    x_beg = x * xstep + nw[0]
                    x_end = x_beg + window[0]
                    y_beg = y * ystep + nw[1]
                    y_end = y_beg + window[1]
                    yield ((x_beg, y_beg), (x_end, y_end))
