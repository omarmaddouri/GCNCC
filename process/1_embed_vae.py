"""
Much of this implementation is inspired by:
https://github.com/greenelab/tybalt/blob/master/scripts/nbconverted/tybalt_vae.py
"""
from __future__ import division
from __future__ import print_function

from pathlib import Path
import sys
project_path = Path(__file__).resolve().parents[1]
sys.path.append(str(project_path))

import numpy as np
import tensorflow as tf
import os
from sklearn.decomposition import KernelPCA
from core.utils import *
import argparse
import pandas as pd
from keras.layers import Input, Dense, Lambda, Layer, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
import tensorflow.keras.backend as K
from keras import metrics, optimizers
from keras.callbacks import Callback

# Set random seed
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--learning_rate',
                    help='learning rate of the optimizer',
                    default=5*10**-4)
parser.add_argument('-b', '--batch_size',
                    help='Number of samples to include in each learning batch',
                    default=100)
parser.add_argument('-e', '--epochs',
                    help='How many times to cycle through the full dataset',
                    default=1000)
parser.add_argument('-k', '--kappa',
                    help='How fast to linearly ramp up KL loss',
                    default=1)
parser.add_argument('-d', '--depth', default=1,
                    help='Number of layers between input and latent layer')
parser.add_argument('-c', '--first_layer',
                    help='Dimensionality of the first hidden layer',
                    default=32)
parser.add_argument('-f', '--dataset',
                    help='Dataset string',
                    default='brc_microarray_usa')
parser.add_argument('-m', '--embedding_method',
                    help='Name of the embedding method',
                    default='vae')
parser.add_argument('-n', '--embedding_size', default=16,
                    help='The latent space dimensionality')
args = parser.parse_args()

# Set hyper parameters
learning_rate = float(args.learning_rate)
batch_size = int(args.batch_size)
epochs = int(args.epochs)
kappa = float(args.kappa)
depth = int(args.depth)
first_layer = int(args.first_layer)
dataset = args.dataset
embedding_method = args.embedding_method
latent_dim = int(args.embedding_size)

#Check dataset availability
if not os.path.isdir("{}/data/parsed_input/{}".format(project_path, dataset)):
    sys.exit("{} dataset is not available under data/parsed_input/".format(dataset))
    
if not os.path.isdir("{}/data/output/{}/embedding/{}".format(project_path, dataset, embedding_method)):
    os.makedirs("{}/data/output/{}/embedding/{}".format(project_path, dataset, embedding_method))

print("--------------------------------------------")
print("--------------------------------------------")
print("Hyper-parameters:")
print("Dataset: {}".format(dataset))
print("Embedding method: {}".format(embedding_method))
print("Dimension of latent space: {}".format(latent_dim))
print("--------------------------------------------")
print("--------------------------------------------")    
# Load Data
X, _, _ = load_training_data(dataset=dataset)
gene_express_df = pd.DataFrame(X)

# Set architecture dimensions
original_dim = gene_express_df.shape[1]
epsilon_std = 1.0
beta = K.variable(0)
if depth == 2:
    latent_dim2 = int(first_layer)

# Random seed
#seed = int(np.random.randint(low=0, high=10000, size=1))
seed = 123
np.random.seed(seed)


# Function for reparameterization trick to make model differentiable
def sampling(args):

    # Function with args required for Keras Lambda function
    z_mean, z_log_var = args

    # Draw epsilon of the same shape from a standard normal distribution
    epsilon = K.random_normal(shape=tf.shape(z_mean), mean=0.,
                              stddev=epsilon_std)

    # The latent vector is non-deterministic and differentiable
    # in respect to z_mean and z_log_var
    z = z_mean + K.exp(z_log_var / 2) * epsilon
    return z


class CustomVariationalLayer(Layer):
    """
    Define a custom layer that learns and performs the training
    """
    def __init__(self, **kwargs):
        # https://keras.io/layers/writing-your-own-keras-layers/
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x_input, x_decoded):
        reconstruction_loss = original_dim * \
                              metrics.categorical_crossentropy(x_input, x_decoded)
        kl_loss = - 0.5 * K.sum(1 + z_log_var_encoded -
                                K.square(z_mean_encoded) -
                                K.exp(z_log_var_encoded), axis=-1)
        return K.mean(reconstruction_loss + (K.get_value(beta) * kl_loss))

    def call(self, inputs):
        x = inputs[0]
        x_decoded = inputs[1]
        loss = self.vae_loss(x, x_decoded)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x


class WarmUpCallback(Callback):
    def __init__(self, beta, kappa):
        self.beta = beta
        self.kappa = kappa

    # Behavior on each epoch
    def on_epoch_end(self, epoch, logs={}):
        if K.get_value(self.beta) <= 1:
            K.set_value(self.beta, K.get_value(self.beta) + self.kappa)

# Process data

# Split 10% test set randomly
test_set_percent = 0.1
gene_express_test_df = gene_express_df.sample(frac=test_set_percent)
gene_express_train_df = gene_express_df.drop(gene_express_test_df.index)

# Input place holder for gene_express data with specific input size
gene_express_input = Input(shape=(original_dim, ))

# ~~~~~~~~~~~~~~~~~~~~~~
# ENCODER
# ~~~~~~~~~~~~~~~~~~~~~~
# Depending on the depth of the model, the input is eventually compressed into
# a mean and log variance vector of prespecified size. Each layer is
# initialized with glorot uniform weights and each step (dense connections,
# batch norm,and relu activation) are funneled separately
#
# Each vector of length `latent_dim` are connected to the gene_express input tensor
# In the case of a depth 2 architecture, input_dim -> latent_dim -> latent_dim2

if depth == 1:
    z_shape = latent_dim
    z_mean_dense = Dense(latent_dim,
                         kernel_initializer='glorot_uniform')(gene_express_input)
    z_log_var_dense = Dense(latent_dim,
                            kernel_initializer='glorot_uniform')(gene_express_input)
elif depth == 2:
    z_shape = latent_dim2
    hidden_dense = Dense(latent_dim,
                         kernel_initializer='glorot_uniform')(gene_express_input)
    hidden_dense_batchnorm = BatchNormalization()(hidden_dense)
    hidden_enc = Activation('relu')(hidden_dense_batchnorm)

    z_mean_dense = Dense(latent_dim2,
                         kernel_initializer='glorot_uniform')(hidden_enc)
    z_log_var_dense = Dense(latent_dim2,
                            kernel_initializer='glorot_uniform')(hidden_enc)

z_mean_dense_batchnorm = BatchNormalization()(z_mean_dense)
z_mean_encoded = Activation('relu')(z_mean_dense_batchnorm)

z_log_var_dense_batchnorm = BatchNormalization()(z_log_var_dense)
z_log_var_encoded = Activation('relu')(z_log_var_dense_batchnorm)

# return the encoded and randomly sampled z vector
# Takes two keras layers as input to the custom sampling function layer with a
# latent_dim` output
z = Lambda(sampling,
           output_shape=(z_shape, ))([z_mean_encoded, z_log_var_encoded])

# ~~~~~~~~~~~~~~~~~~~~~~
# DECODER
# ~~~~~~~~~~~~~~~~~~~~~~
# The layers are different depending on the prespecified depth.
#
# Single layer: glorot uniform initialized and sigmoid activation.
# Double layer: relu activated hidden layer followed by sigmoid reconstruction
if depth == 1:
    decoder_to_reconstruct = Dense(original_dim,
                                   kernel_initializer='glorot_uniform',
                                   activation='sigmoid')
elif depth == 2:
    decoder_to_reconstruct = Sequential()
    decoder_to_reconstruct.add(Dense(latent_dim,
                                     kernel_initializer='glorot_uniform',
                                     activation='relu',
                                     input_dim=latent_dim2))
    decoder_to_reconstruct.add(Dense(original_dim,
                                     kernel_initializer='glorot_uniform',
                                     activation='sigmoid'))

gene_express_reconstruct = decoder_to_reconstruct(z)

# ~~~~~~~~~~~~~~~~~~~~~~
# CONNECTIONS
# ~~~~~~~~~~~~~~~~~~~~~~
adam = optimizers.Adam(lr=learning_rate)
vae_layer = CustomVariationalLayer()([gene_express_input, gene_express_reconstruct])
vae = Model(gene_express_input, vae_layer)
vae.compile(optimizer=adam, loss=None, loss_weights=[beta])

# fit Model
hist = vae.fit(np.array(gene_express_train_df),
               shuffle=True,
               epochs=epochs,
               batch_size=batch_size,
               validation_data=(np.array(gene_express_test_df), None),
               callbacks=[WarmUpCallback(beta, kappa)])


encoder = Model(gene_express_input, z_mean_encoded)
embeddings = encoder.predict_on_batch(gene_express_df)
#Save the node emmbeddings
np.savetxt("{}/data/output/{}/embedding/{}/embeddings.txt".format(project_path, dataset, embedding_method), embeddings, delimiter="\t")
print("Embeddings saved in /data/output/{}/embedding/{}/embeddings.txt".format(dataset, embedding_method))