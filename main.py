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

def load_dataset(feat, **kwargs):
    v_ims = common.load_images("data/vehicles/", "data/OwnCollection/vehicles/", color='HSV')
    nv_ims = common.load_images("data/non-vehicles/", "data/OwnCollection/non-vehicles/", color='HSV')
    return feat.extract(v_ims, **kwargs), feat.extract(nv_ims, **kwargs)

def fit_and_normalize(feat, x, mode='standard'):
    feat.fit_scaler(x, mode)
    return feat.normalize(x)

def save_dataset_norm(x, y, filename='data/dataset-norm.p'):
    save_dataset(x, y, filename=filename)

def save_dataset(x, y, filename='data/dataset.p'):
    with open(filename, 'wb') as fd:
        pickle.dump({'x': x, 'y': y}, fd)

def dataset_norm():
    return load_dataset(filename='data/dataset-norm.p', onlyfile=True)
    
def dataset(filename='data/dataset.p', onlyfile=False):
    f = Features()
    if os.path.isfile(filename):
        with open(filename, 'rb') as fd:
            data = pickle.load(fd)
            x, y = data['x'], data['y']
    elif onlyfile == False:
        v_feat, nv_feat = load_dataset(f)
        v_lbl = np.array([1] * v_feat.shape[0])
        nv_lbl = np.array([0] * nv_feat.shape[0])
        x = np.vstack((v_feat, nv_feat))
        y = np.concatenate([v_lbl, nv_lbl])
    else:
        return f, None, None
    return f, x, y

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

def split_data(x, y, random_state=101):
    xtr, xt, ytr, yt = train_test_split(x, y, test_size=0.2, random_state=random_state)
    return (xtr, ytr), (xt, yt)

def train_xgboost(x, y, xtest, ytest, show=False, random_state=101):
    xgb = XGBClassifier(
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
    xgb.fit(x, y, eval_metric='auc')
    pred = xgb.predict(xtest)
    pred_prob = xgb.predict_proba(xtest)
    acc_msg = "Test accuracy: {0:.05f}"
    auc_msg = "AUC score: {0:.05f}"
    importance = pd.Series(xgb.booster().get_fscore()).sort_values(ascending=False)
    importance.plot(kind='bar', title='Feature Importance')
    plt.show()
    print(acc_msg.format(metrics.accuracy_score(ytest, pred)))
    print(pred[:10])
    print(pred_prob[:10])
    print(auc_msg.format(metrics.roc_auc_score(ytest, pred_prob)))
    return xgb
  
def embeddings(x, y):
    LOG_DIR = 'tflog'
    embedding_var = tf.Variable(xnorm, name='vehicles')
    step = tf.Variable(0, trainable=False, name='global_step')
    with open(metadata, 'wb') as metadata_file:
        for row in y:                           
            metadata_file.write(bytes('%d\n' % row, 'utf-8'))
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), step)
        summary_writer = tf.summary.FileWriter(LOG_DIR)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
        projector.visualize_embeddings(summary_writer, config)

