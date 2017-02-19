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
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import importlib

importlib.reload(common)

from common import show_image

class Features:
    def __init__(self):
        self._scaler = None
    def fit_scaler(self, x, mode='standard'):
        """ Initializes scaler.
        Args:
            x: input feature vectors
            mode: 'standard' or 'minmax'
        """
        if mode == 'standard':
            self._scaler = StandardScaler().fit(x)
        elif mode == 'minmax':
            self._scaler = MinMaxScaler(feature_range=(-1,1)).fit(x)
        else:
            raise ValueError('Wrong mode passed as argument')
    def normalize(self, x):
        if self._scaler is None:
            raise ValueError("Scaler is not initialized")
        return self._scaler.transform(x)
    def extract(self, ims, **kwargs):
        """ Extract features from images with three channels.
        """
        n = ims.shape[0]
        cls = self.__class__
        return np.array([cls._extract_features(ims[i], **kwargs) for i in range(n)],
                        dtype=np.float32)
    @classmethod
    def _extract_features(cls, im,
               inc_hog_channel='all', inc_color_hist=True,
               inc_spatial_bins=True, orients=8,
               cell_size=7, block_size=2,
               vector=True, dst_size=16,
               bins=32, binrange=(0,256),
               show=False):
        fhog, fcolor, fspatial = [], [], []
        if inc_hog_channel != -1:
            num_chan = im.shape[2]
            if inc_hog_channel.lower() == 'all':
                fhog = np.ravel([cls.hog_features(im[:,:,i], orients, cell_size, block_size, vector)
                                 for i in range(num_chan)])
            else:
                i = inc_hog_channel
                if inc_hog_channel >= num_chan:
                    raise ValueError("Access to non-existing channel {0}".format(i))
                fhog = cls.hog_features(im[:,:,i], orients, cell_size, block_size, vector)
        if inc_color_hist:
            fcolor = cls.color_hist_features(im, bins, binrange)
        if inc_spatial_bins:
            fspatial = cls.binspatial_features(im, dst_size)
        return np.concatenate([np.float32(fhog), np.float32(fcolor), np.float32(fspatial)])
    @staticmethod
    def hog_features(im, orients=8, cell_size=8, block_size=2, vector=True, show=False):
        if show == True:
            features, im_hog = hog(im, orientations=orients,
                                   pixels_per_cell=(cell_size,)*2,
                                   cells_per_block=(block_size,)*2,
                                   feature_vector=vector,
                                   transform_sqrt=True,
                                   visualise=show)
            show_image([im, im_hog], ncols=2,
                       window_title='HOG',
                       titles=['original', 'hog'],
                       cmaps=['gray', 'gray'])
        else:
            features = hog(im, orientations=orients,
                           pixels_per_cell=(cell_size,)*2,
                           cells_per_block=(block_size,)*2,
                           feature_vector=vector,
                           transform_sqrt=True,
                           visualise=show)
        return features
    @staticmethod
    def binspatial_features(im, dst_size=16, show=False):
        features = cv.resize(im, (dst_size,)*2)
        if show == True:
            show_image([im, features], ncols=2,
                       window_title='Bin Spatial',
                       titles=['original', 'resized'])
        return features.ravel()
    @staticmethod
    def color_hist_features(im, bins=32, binrange=(0,256), show=False):
        chan1 = np.histogram(im[:,:,0], bins=bins, range=binrange)[0]
        chan2 = np.histogram(im[:,:,1], bins=bins, range=binrange)[0]
        chan3 = np.histogram(im[:,:,2], bins=bins, range=binrange)[0]
        if show == True:
            fig, axes = plt.subplots(nrows=4, squeeze=True)
            fig.canvas.set_window_title('Histograms')
            x = range(chan1.shape[0])
            axes[0].imshow(im)
            axes[0].set_title('image')
            axes[0].get_xaxis().set_visible(False)
            axes[0].get_yaxis().set_visible(False)
            axes[1].hist(x, chan1)
            axes[1].set_title('chan_1')
            axes[2].hist(x, chan1)
            axes[2].set_title('chan_2')
            axes[3].hist(x, chan1)
            axes[3].set_title('chan_3')
            fig.tight_layout()
            fig.subplots_adjust(wspace=0, left=-0.1)
        return np.concatenate([chan1, chan2, chan3])
