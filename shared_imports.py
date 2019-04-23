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
from sklearn.neighbors import NearestNeighbors

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

# returns a random sample of indices from the full set
def random_subsampled_indices(full_index_no, subsample_no):
    index_list = list(range(0, full_index_no))
    random.shuffle(index_list)
    index_list = index_list[0:subsample_no]
    return index_list

# loads and preprocesses MNIST data
# X_train is (60000 x 28 x 28)
# X_test is (10000 x 28 x 28)
# y_train is (60000 x 1)
# x_train is (10000 x 1)
def load_mnist(subsample_no=0):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # this normalization seems necessary or traditional for (grayscale) images
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # vectorize the pixels of the input images
    # this takes X_train from (?, 28, 28) to (?, 784)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])

    if subsample_no > 0:
        index_list = random_subsampled_indices(X_train.shape[0], subsample_no)
        X_train = X_train[index_list]
        y_train = y_train[index_list]
        # this assumes subsample_no >= min(X_train.shape[0], X_test.shape[0])
        index_list = random_subsampled_indices(X_test.shape[0], subsample_no)
        X_test = X_test[index_list]
        y_test = y_test[index_list]

    # center the data; this seems appropriate for cosine angle, probably doesn't
    # hurt for other measure
    colMeans = X_train.mean(axis=0)
    X_train = X_train - colMeans
    return X_train, y_train, X_test, y_test

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
    # e.g. kernel = RBF() + WhiteKernel()
    # seems to work better in this case with a linear kernel?
    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X_train, y_train)
    if return_sigma:
        return gpr.predict(X_test, return_std=True)
    else:
        return gpr.predict(X_test)

def predict_CNN_regression(X_train, y_train, X_test, batch_size=10, epochs=10, verbose=0):
    n = X_train.shape[0] # may varying if subsetting for debugging
    cnn = KerasRegressor(build_fn=regression_model, epochs=epochs, batch_size=batch_size, verbose=verbose)
    cnn.fit(x=X_train.reshape(n, 3, 1, 1), y=y_train.reshape(n, 1))
    return cnn.predict(X_test.reshape(1, 3, 1, 1))

# takes a similarity or distance matrix and returns the vector of upper triangular measurements
def vectorize_measure(csim):
    csim_vec = list()
    no_samples = csim.shape[0]
    for i in range(no_samples):
        for j in range((i+1), no_samples):
            csim_vec.append(csim[i,j])
    return csim_vec

# wrapper
def measure(X, sim_dist_method="cosine"):
    csim = None
    if sim_dist_method == "Euclidean":
        csim = euclidean_distances(X)
    elif sim_dist_method == "KNN":
        nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(X)
        csim = nbrs.kneighbors_graph(X).toarray()
    else:
        csim = cosine_similarity(X)
    return csim

# perform an embedding of X
def embed(X, n_components, embed_method="PCA"):
    X_transformed = None
    if embed_method == "Laplacian eigenmaps":
        embedding = SpectralEmbedding(n_components=n_components)
        X_transformed = embedding.fit_transform(X)
    else:
        pca = decomposition.PCA(n_components=n_components)
        pca.fit(X)
        X_transformed = pca.transform(X)
    return X_transformed

# wrapper to embed and get the vector of (off-diagonal) pairwise distances
# here csim_hd is the similarity or distance matrix over the same samples in the HD space
def embed_and_correlate(X, n_components, csim_hd, embed_method="PCA", sim_dist_method="cosine"):
    X_transformed = embed(X, n_components, embed_method=embed_method)
    csim_ld_vec = None
    mean_measure = 0.0
    std_measure = 0.0
    corr_measure = 0.0
    csim_ld = measure(X_transformed, sim_dist_method=sim_dist_method)
    if sim_dist_method == "KNN":
        csim_hd_vec = vectorize_measure(csim_hd)
        csim_ld_vec = vectorize_measure(csim_ld)
        logical_int = np.logical_and(csim_hd_vec, csim_ld_vec)
        corr_measure = sum(logical_int) / sum(csim_hd_vec) # proportion of neighbors retained
    else:
        # vectorize and correlate
        csim_ld_vec = vectorize_measure(csim_ld)
        mean_measure = np.mean(csim_ld_vec)
        std_measure = np.std(csim_ld_vec)
        csim_hd_vec = vectorize_measure(csim_hd)
        corr_measure = np.corrcoef(csim_hd_vec, csim_ld_vec)[0,1]
    return (X_transformed, mean_measure, std_measure, corr_measure)

# heavily borrowing architecture from: https://keras.io/examples/variational_autoencoder/
# network parameters
def vae_embed(X_train, X_test, latent_dim=3, epochs=10, verbose=1):
    # we'll only use this with MNIST; hard-code the dimensions
    original_dim = 28*28
    input_shape = (original_dim, )
    intermediate_dim = 12*12 # pretty arbitrary

    # encoder; simple - no convolution, just dense layers
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # define a sample vector from the latent distribution
    # this will be the input into the decoder!
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    decoder = Model(latent_inputs, outputs, name='decoder')

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')
    models = (encoder, decoder)

    reconstruction_loss = mse(inputs, outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    vae.fit(X_train, epochs=epochs, batch_size=128, validation_data=(X_test, None), verbose=verbose)
    
    # just return the fitted /encoder/, since all we want to do is predict
    return encoder


















