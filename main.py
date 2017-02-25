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
import track
import model

import importlib
from calibrator import Calibrator as clb
from moviepy.editor import VideoFileClip

importlib.reload(track)
importlib.reload(model)

def single_image(filename, model_path='./data/model.p'):
    if not clb.initialized():
        clb.find_pictures(directory='./camera_cal/')
        clb.calibrate_camera(9, 6)
    im = common.load_image(filename, color='RGB')
    common.show_image(im)
    m = model.CarModel()
    m.load(filename=model_path)
    t = track.FrameVehiclePipeline(m, shape=im.shape[:2])
    t.process(im, show=True)

def single_video(filename, output='./output.mp4'):
    if not clb.initialized():
        clb.find_pictures(directory='./camera_cal/')
        clb.calibrate_camera(9, 6)
    m = model.CarModel()
    m.load()
    t = track.FrameVehiclePipeline(m, shape=(720,1280))
    video = VideoFileClip(filename)
    out = video.fl_image(t.process)
    out.write_videofile(output, audio=False)

def generate_samples(filename, output='./output.mp4'):
    if not clb.initialized():
        clb.find_pictures(directory='./camera_cal/')
        clb.calibrate_camera(9, 6)
    m = model.CarModel()
    m.load()
    t = track.FrameVehiclePipeline(m, shape=(720,1280))
    video = VideoFileClip(filename)
    out = video.fl_image(t.samples)
    out.write_videofile(output, audio=False)

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

