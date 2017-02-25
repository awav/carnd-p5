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

from tensorflow.contrib.tensorboard.plugins import projector
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from xgboost.sklearn import XGBClassifier

import importlib

importlib.reload(features)
importlib.reload(common)

from features import Features

class VehiclesDataset():
    def __init__(self, load=True,
                 vehicles=["data/vehicles/"],
                 nonvehicles=["data/non-vehicles"],
                 color='HSL'):
        if load == True:
            self.color = color
            veh = common.load_images(*vehicles, color=color)
            nonveh = common.load_images(*nonvehicles, color=color)
            veh_lbl = np.array([1] * veh.shape[0])
            nonveh_lbl = np.array([0] * nonveh.shape[0])
            self.x_orig = np.concatenate([veh, nonveh])
            self.y_orig = np.concatenate([veh_lbl, nonveh_lbl])
        else:
            self.x_orig = None
            self.y_orig = None
        self.x = None
        self.colormap = color
    def put_features(self, features):
        self.x = features
    def embeddings(self, log_dir='tflog'):
        embedding_var = tf.Variable(self.x, name='vehicles')
        step = tf.Variable(0, trainable=False, name='global_step')
        metadata = os.path.join(log_dir, 'metadata.tsv')
        with open(metadata, 'w') as metadata_file:
            for i in self.y_orig:
                if i == 0:
                    lbl = 'non-vehicle'
                elif i == 1:
                    lbl = 'vehicle'
                metadata_file.write(lbl + '\n')
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver()
            saver.save(sess, os.path.join(log_dir, "model.ckpt"), step)
            summary_writer = tf.summary.FileWriter(log_dir)
            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name
            embedding.metadata_path = metadata 
            projector.visualize_embeddings(summary_writer, config)
    def load(self, filename='data/dataset.p'):
        with open(filename, 'rb') as fd:
             self.__dict__ = pickle.load(fd)
    def save(self, filename='data/dataset.p'):
        with open(filename, 'wb') as fd:
             pickle.dump(self.__dict__, fd)

class CarModel():
    def __init__(self, mode='svm'):
        if mode != 'svm' and mode != 'xgboost':
            raise ValueError('Unknown mode for CarModel')
        self._f = Features()
        self._model = None
        self._mode = mode
        self.input_shape = None
    def prepare(self, data, mode='standard'):
        self.input_shape = data.x_orig[0].shape
        features = self._f.extract(data.x_orig)
        self._f.fit_scaler(features, mode=mode)
        x = self._f.normalize(features)
        data.put_features(x)
        self._colormap = data.colormap
        return data
    def fit(self, data, random_state=11, show=False):
        if data.x is None:
            raise ValueError('Dataset does not have input values')
        x = data.x
        y = data.y_orig
        train, test = self._split_data(x, y)
        self._train(train, test, show=show)
    def predict(self, im, show=False):
        f = self._f
        im = common.cvt_color(im, color=self._colormap, src='RGB')
        pred = self._model.predict(f.normalize(f.extract(np.array([im]), show=show)))
        return pred[0]
    def _split_data(self, x, y, test_size=0.1, random_state=11):
        xtr, xt, ytr, yt = train_test_split(x, y, test_size=test_size, random_state=random_state)
        return (xtr, ytr), (xt, yt)
    def _one_hot_encode(self, y):
        width = np.unique(y).shape[0]
        height = y.shape[0]
        one = np.zeros((height, width), dtype=np.int32)
        one[range(height), y] = 1
        return one
    def _train(self, train, test, random_state=11, show=False):
        x, y = train
        xtest, ytest = test
        if self._mode == 'xgboost':
            self._model = XGBClassifier(
                              learning_rate=0.1,
                              n_estimators=150,
                              max_depth=5,
                              min_child_weight=1,
                              gamma=0,
                              subsample=0.8,
                              colsample_bytree=0.8,
                              objective='binary:logistic',
                              nthread=4,
                              scale_pos_weight=1,
                              seed=random_state)
            self._model.fit(x, y, eval_metric='auc')
        else:
            self._model = LinearSVC(random_state=random_state)
            #self._model = SVC(kernel='rbf', max_iter=25000, random_state=random_state)
            #self._model = SVC(max_iter=25000, random_state=random_state)
            self._model.fit(x, y)
        pred = self._model.predict(xtest)
        acc_msg = "Test accuracy: {0:.05f}"
        print(acc_msg.format(metrics.accuracy_score(ytest, pred)))
        if self._mode == 'xgboost':
                pred_prob = self._model.predict_proba(xtest)
                ytest_hot = self._one_hot_encode(ytest)
                auc_msg = "AUC score: {0:.05f}"
                print(auc_msg.format(metrics.roc_auc_score(ytest_hot, pred_prob)))
        if self._mode == 'xgboost' and show == True:
            importance = pd.Series(self._model.booster().get_fscore()).sort_values(ascending=False)
            importance.plot(kind='bar', title='Feature Importance')
            plt.show()
            print(importance[:10])
    def load(self, filename='data/model.p'):
        with open(filename, 'rb') as fd:
             self.__dict__ = pickle.load(fd)
    def save(self, filename='data/model.p'):
        with open(filename, 'wb') as fd:
             pickle.dump(self.__dict__, fd)

