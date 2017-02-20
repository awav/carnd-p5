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
import sys
import os
import re

class Calibrator():
    _files,_mtx, _dist = [], None, None
    def __new__(cls, *args, **kvargs):
        raise ValueError("You can't create `Calibration` instance")
    @classmethod
    def initialized(cls):
        return (cls._mtx is not None) and (cls._dist is not None)
    @classmethod
    def find_pictures(cls, pattern='.*\.jpg', directory='.'):
        if not os.path.isdir(directory):
            ## TOOD: log this message
            return []
        regex = re.compile(pattern)
        files = []
        for filename in os.listdir(directory):
            if re.match(string=filename, pattern=regex) is None:
                continue
            files.append(os.path.join(directory, filename))
        if files != []:
            cls._files = files
        return files
    @classmethod
    def calibrate_camera(cls, nx, ny, show=False):
        assert(len(cls._files) != 0)
        objs = np.zeros((nx * ny, 3), dtype=np.float32)
        objs[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2)
        n = len(cls._files)
        objpoints = [objs] * n
        imgpoints = []
        for i in range(n):
            img = cv.imread(cls._files[i], cv.IMREAD_COLOR)
            if img is None:
                print('{0} is not an image'.format(cls._files[i]), file=sys.stderr)
                continue
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, (nx, ny), None)
            if ret == True:
                imgpoints.append(corners)
                if show == True:
                    cv.drawChessboardCorners(img, (nx, ny), corners, ret)
                    fig = plt.figure(0)
                    fig.canvas.set_window_title('calibrate image #{0}'.format(i))
                    plt.imshow(img)
                    plt.show()
            elif show == True:
                fig = plt.figure(0)
                fig.canvas.set_window_title('FAILED image #{0}'.format(i))
                plt.imshow(img)
                plt.show()
        nimg = len(imgpoints)
        if nimg > 0:
            shape = img.shape[1], img.shape[0]
            ret, mtx, dist, _rv, _tv = cv.calibrateCamera(
                    objpoints[:nimg], imgpoints,
                    imageSize=shape, cameraMatrix=None, distCoeffs=None)
            if ret:
                cls._mtx = mtx
                cls._dist = dist
    @classmethod
    def undistort(cls, im, show=False):
        assert(cls._mtx is not None)
        assert(cls._dist is not None)
        undist_im = cv.undistort(im, cls._mtx, cls._dist, None, cls._mtx)
        if show == True:
            fig, ax = plt.subplots(ncols=2, squeeze=True)
            fig.canvas.set_window_title('Distorted/Undistored Image')
            ax[0].imshow(im)
            ax[1].imshow(undist_im)
            ax[0].set_title('distorted', color='r')
            ax[1].set_title('undistorted', color='b')
            fig.tight_layout()
            fig.show()
        return undist_im
