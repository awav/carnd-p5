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

import tensorflow as tf
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import common
import features
import pickle
import os

from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier

import importlib

importlib.reload(features)
importlib.reload(common)

from features import Features

def visualize():
    print("Load images")
    v_ims1 = common.load_images("data/vehicles/", color='RGB')
    v_ims2 = common.load_images("data/OwnCollection/vehicles/", color='RGB')
    nv_ims1 = common.load_images("data/non-vehicles/", color='RGB')
    nv_ims2 = common.load_images("data/OwnCollection/non-vehicles/", color='RGB')
    ncols, nrows = (16,) * 2
    n = ncols * nrows // 4
    ims = [v_ims1[np.random.randint(v_ims1.shape[0], size=n)],
           v_ims2[np.random.randint(v_ims2.shape[0], size=n)],
           nv_ims1[np.random.randint(nv_ims1.shape[0], size=n)],
           nv_ims2[np.random.randint(nv_ims2.shape[0], size=n)]]
    print("Load images done")
    ims = np.concatenate(ims)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows)
    fig.canvas.set_window_title('Dataset images')
    print("Visualize images {0}x{0}".format(ncols))
    size = ims.shape[0]
    for r, row_ax in enumerate(axes):
        for c, ax in enumerate(row_ax):
            i = r * ncols + c
            ax.imshow(ims[i])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            #ax.set_aspect('equal')
    fig.subplots_adjust(wspace=0, hspace=0)
    print("Show images")
    fig.show()

import model

def rects(img, factor_w=0.9, factor_h=1.5):
    im = img.copy()
    y = 400
    h = 60
    w = 10
    sh, sw, _ = im.shape
    print(sh, sw)
    for i in range(1):
        width = h * w
        topx = (sw - width) // 2
        botx = sw - topx
        topy = y
        boty = topy + h
        h = np.int32(factor_h * h)
        w = np.int32(factor_w * w)
        print((topx, topy),(botx, boty))
        im = cv.rectangle(im, (topx, topy), (botx, boty), (255,5*i+1,0), 1)
    return im
