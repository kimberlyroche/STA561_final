import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# suppress tensorflow info/warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import time

import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
# %matplotlib inline # for Jupyter
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pickle

# for GP regression
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import OneHotEncoder

# for CNN
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from keras import backend as K
from keras.preprocessing import image
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Layer, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, Activation, Lambda, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import mse, categorical_crossentropy

# for spectral embedding
from keras.datasets import mnist, cifar100, fashion_mnist
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances, rbf_kernel
from sklearn import decomposition

round_tol = 6

# =======================================================================================================
#     SHARED FUNTIONS
# =======================================================================================================

# sources for CNN design:
# https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
# https://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/

# loads and preprocesses data from lab 1
def load_lab1_data(subsample_no = 0):
    data = pd.read_csv("data/Advertisement.csv", header=None, skiprows=1, \
        names=["ID", "TV", "radio", "newspaper", "sales"])
    no_samples = data.shape[0]

    X = data[['TV', 'radio', 'newspaper']]
    y = data[['sales']]

    if subsample_no > 0:
        X = X[0:subsample_no]
        y = y[0:subsample_no]

    # various internet sources suggest best performance when standardizing data for input
    # into neural networks, so we'll do that ahead of time (and for GP regression too)
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    y = min_max_scaler.fit_transform(y)

    return (X, y)

# dead-simple convolutional architecture for the simple 3 x 1 regression problem
def regression_model():
    input_img = Input(shape=(3, 1, 1))
    x = Conv2D(2, (2, 1), activation='relu', padding='same')(input_img)
    x = Flatten()(x)
    x = Dense(1)(x)
    model = Model(input_img, x)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# both predict_* methods expect np.arrays of dimensions
# X_train: (no_samples-1, no_features)
# y_train: (no_samples-1, dim_outcome)
# X_test: (1, no_features)
# y_test: (1, dim_outcome)

def predict_GP(X_train, y_train, X_test, return_sigma=False):
    # lots of possible choices for kernel here!
    kernel = RBF() + WhiteKernel()
    #kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X_train, y_train)
    if return_sigma:
        return gpr.predict(X_test, return_std=True)
    else:
        return gpr.predict(X_test)

def predict_CNN_regression(X_train, y_train, X_test, batch_size=10, epochs=10):
    n = X_train.shape[0] # may varying if subsetting for debugging
    cnn = KerasRegressor(build_fn=regression_model, epochs=epochs, batch_size=batch_size, verbose=0)
    cnn.fit(x=X_train.reshape(n, 3, 1, 1), y=y_train.reshape(n, 1))
    return cnn.predict(X_test.reshape(1, 3, 1, 1))