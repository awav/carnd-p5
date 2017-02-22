# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### Feature extraction
#### Criteria: explain how (and identify where in your code) you extracted HOG features from the training images. Explain how you settled on your final choice of HOG parameters.

I used data provided by Udacity for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) dataset. I started with code that loads this datasets and represents it in convenient way for futher steps. So, I wrote a class which loads data and keeps `X`s and `Y`s for futher model training. As you can see I can save and load an instace of this class for later usage. It is very helpful when you are playing with different datasets.

```python
### model.py
ass VehiclesDataset():
    def __init__(self, load=True,
                 #vehicles=["data/vehicles/", "data/OwnCollection/vehicles/"],
                 #nonvehicles=["data/non-vehicles/", "data/OwnCollection/non-vehicles/"],
                 vehicles=["data/vehicles/"],
                 nonvehicles=["data/non-vehicles/"],
                 color='HSV'):
        if load == True:
            self.color = color
            veh = common.load_images(*vehicles, color=color)
            nonveh = common.load_images(*nonvehicles, color=color)
            veh_lbl = np.array([1] * veh.shape[0])
            nonveh_lbl = np.array([0] * nonveh.shape[0])
            self.x_orig = np.vstack([veh, nonveh])
            self.y_orig = np.concatenate([veh_lbl, nonveh_lbl])
        else:
            self.x_orig = None
            self.y_orig = None
        self.x = None
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
```

To extract features I used stadard solution basically provided by Udacity in the course. I modified a bit interface and optimized the way how they are computed for whole dataset.

I used HOG features, spatial binning and color histograms altogether. All input images are converted into HSV color scheme before feature extracting. Let's take a look at default arguments and paramerters for these feature extractors:

* Hog params:
   - use all channels of an inputs image
   - number of orientations is 8
   - cell_size is 
   - block_size is 2
* Image bins params:
   - number of bins is 32
* Color histograms params:
   - number of bins is 32

In fact, I deliberaterly tried to increase number of features, because of classifcation models which I trained to solve this challenge. It was very important to to have a lot of features as input for linear SVM and xgboost.

```python
## features.py
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
               cell_size=8, block_size=2,
               vector=True, dst_size=32,
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
```

To understand the data, it is very helpful to find embeddings for the dataset. I used standard approaches which help to squash features into lower dimentions, so that the dataset could be visualized in 3D or 2D dimension. If you look at PCA and t-SNE gifs you will find orange and blue clouds of dots are more or less separable. The orange and blue dots represent `non-vehicle` and `vehicle` respectively. PCA gives quite poor result though, but t-SNE clearly makes great job. It shows that updated dataset has patterns: there are sub-clusters in vehicles and one big blob of non-vehicles.

PCA applied to extracted feature dataset (including HOGs, spatial bins, color histogram):
![alt text](project/pca.gif)

t-SNE applied to extracted feature dataset (including HOGs, spatial bins, color histogram) with `perplexity=80` and `epsilon=10`, known as learning rate as well:
![alt text](project/tsne.gif)

The code for embeddings you can find in `VehicleDataset` class.

### Model training
#### Criteria: The HOG features extracted from the training data have been used to train a classifier, could be SVM, Decision Tree or other. Features should be scaled to zero mean and unit variance before training the classifier.

I inclined to use `XGBoost` to classify vehicles, but during testing I found that `SVM` is much faster in training and in value predicting. Also, `SVM` are more stable despite the fact that `XGBoost` was getting better accuracy on testing data. Comparing of `LinearSVM` with `RBF-SVM` didn't lead to happy results, the reason for that is huge number of features. Usually, `RBF-SVM` gets better results when number of dataset samples is bigger than number of predictors, but in our case we have enough degrees of freedom to build robust classifier based on `Linear SVM`.

The one benefit that I got from using XGBoost is that I got a distribution of importance scores for features. All top 10 values are HOG features. It means that HOG values contribute a lot to the final decision on inputs.

#### XGBoost resuls

```
Test accuracy: 0.99747
AUC score: 0.99994
```

| Feature index | Importance score |
|---------------|------------------|
|  4715         | 93               |             
|  4726         | 80               |
|  4767         | 74               |
|  4799         | 39               |
|  4798         | 31               |
|  4797         | 26               |
|  4775         | 24               |
|  4706         | 22               |
|  4713         | 22               |
|  3393         | 20               |

#### SVM results

```
Test accuracy: 0.99747
```

The code below summarizes training and predicting processes for chosen model. This class can be saved and loaded by demand to and from disk respectively.

As you can see training consists of three steps:

1. Splitting data into training and testing subsets
2. Training model, depending on mode (`XGBoost` or `SVM`)
3. Getting an accuracy of the model

```python
## model.py
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
        return data
    def fit(self, data, random_state=101, show=True):
        if data.x is None:
            raise ValueError('Dataset does not have input values')
        x = data.x
        y = data.y_orig
        train, test = self._split_data(x, y)
        self._train(train, test, show=show)
    def predict(self, im, show=False):
        f = self._f
        pred = self._model.predict(f.normalize(f.extract(np.array([im]), show=show)))
        return pred[0]
    def _split_data(self, x, y, test_size=0.2, random_state=101):
        xtr, xt, ytr, yt = train_test_split(x, y, test_size=test_size, random_state=random_state)
        return (xtr, ytr), (xt, yt)
    def _one_hot_encode(self, y):
        width = np.unique(y).shape[0]
        height = y.shape[0]
        one = np.zeros((height, width), dtype=np.int32)
        one[range(height), y] = 1
        return one
    def _train(self, train, test, random_state=101, show=False):
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
            self._model = LinearSVC(max_iter=25000, penalty='l2', random_state=random_state)
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

```

