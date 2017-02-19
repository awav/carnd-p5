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

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import re

def load_image(filename, color='RGB'):
    colors = {'RGB':cv.COLOR_BGR2RGB,
              'HLS':cv.COLOR_BGR2HLS,
              'HSV':cv.COLOR_BGR2HSV,
              'LUV':cv.COLOR_BGR2LUV,
              'YUV':cv.COLOR_BGR2YUV,
              'YCrCb':cv.COLOR_BGR2YCrCb}
    if color not in colors:
        raise ValueError("{0} color is not allowed".format(color))
    im = cv.imread(filename, cv.IMREAD_COLOR)
    return cv.cvtColor(im, colors[color.upper()])

def list_images(folder, regex=None):
    return np.array([filename for filename in _list_images(folder)])

def _list_images(folder, regex=None):
    """
    Generator to list images in directory. Default regex parameter is None and
    it will search for png and jpg images.
    """
    if not os.path.isdir(folder):
        ## TOOD: log this message
        return
    folder = os.path.normpath(folder)
    folder_len = len(folder)
    default_pattern = '.*\.(png|jpg|jpeg|PNG|JPG|JPEG)$'
    regex = default_pattern if regex == None else regex
    regex_compiled = re.compile(regex)
    for subdir, mid, files in os.walk(folder):
        subdir = os.path.normpath(subdir)
        if subdir[:folder_len] != folder:
            continue
        if  files == []:
            continue
        for filename in files:
            if re.match(regex_compiled, filename) is None:
                continue
            yield os.path.join(subdir, filename)

def load_images(*dirs, color='RGB'):
    return np.array([im for im in _load_images(*dirs, color=color)])
    
def _load_images(*dirs, color):
    """
    Generator to list images in specified dirs.
    By default it loads images in RGB.
    """
    subdirs = set()
    for folder in dirs:
        for filename in _list_images(folder):
            subdirname = os.path.dirname(filename)
            if subdirname not in subdirs:
                subdirs.add(subdirname)
                print(subdirname)
            yield load_image(filename, color)

def serialize(obj, filename):
    with open(filename, 'wb') as fd:
        pickle.dump(obj, fd)

def show_image(ims, ncols=1, nrows=1, window_title=None, titles=None, cmaps=None):
    """
    Show images as tiles in grid.
    """
    print(ncols, nrows)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, squeeze=True)
    if window_title is not None:
        fig.canvas.set_window_title(window_title)
    size = len(ims)
    if ncols == 1 or nrows == 1:
        n = max(ncols, nrows)
        for i in range(n):
            ax = axes[i]
            if cmaps is not None:
                ax.imshow(ims[i], cmap=cmaps[i])
            else:
                ax.imshow(ims[i])
            if titles is not None:
                ax.set_title(titles[i])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    else:
        for r in range(nrows):
            for c in range(ncols):
                ax = axes[r][c]
                i = r * ncols + c
                if i >= size:
                    ax.axis('off')
                    continue
                if cmaps is not None:
                    ax.imshow(ims[i], cmap=cmaps[i])
                else:
                    ax.imshow(ims[i])
                if titles is not None:
                    ax.set_title(titles[i])
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0)
    fig.show()

def equalize_hist(im, show=False):
    im_eq = cv.cvtColor(im, cv.COLOR_RGB2YCrCb)
    zeros = np.zeros(im_eq.shape[:2])
    im_eq[:,:,0] = cv.equalizeHist(im_eq[:,:,0], zeros)
    im_eq = cv.cvtColor(im_eq, cv.COLOR_YCrCb2RGB)
    if show == True:
        show_images(im, im_eq, 'original', 'equalized', 'Histogram equalization')
    return im_eq
