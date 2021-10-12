from __future__ import division
from __future__ import print_function

from pathlib import Path
import sys
project_path = Path(__file__).resolve().parents[1]
sys.path.append(str(project_path))

from keras.layers import Dense, Activation, Dropout
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K
import numpy as np
import time
import tensorflow as tf
import os

from core.utils import *
from core.layers.graph_cnn_layer import GraphCNN
from sklearn.preprocessing import normalize


# Set random seed
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)

# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'brc_microarray_usa', 'Dataset string.')
flags.DEFINE_string('embedding_method', 'ge', 'Name of the embedding method.')

#Check dataset availability
if not os.path.isdir("{}/data/parsed_input/{}".format(project_path, FLAGS.dataset)):
    sys.exit("{} dataset is not available under data/parsed_input/".format(FLAGS.dataset))
    
if not os.path.isdir("{}/data/output/{}/embedding/{}".format(project_path, FLAGS.dataset, FLAGS.embedding_method)):
    os.makedirs("{}/data/output/{}/embedding/{}".format(project_path, FLAGS.dataset, FLAGS.embedding_method))

print("--------------------------------------------")
print("--------------------------------------------")
print("Hyper-parameters:")
print("Dataset: {}".format(FLAGS.dataset))
print("Embedding method: {}".format(FLAGS.embedding_method))
print("--------------------------------------------")
print("--------------------------------------------")

# Prepare Data
X, A, Y = load_training_data(dataset=FLAGS.dataset)

Y_train, Y_val, Y_test, train_idx, val_idx, test_idx, train_mask = get_splits_for_learning(Y, dataset=FLAGS.dataset)

# Normalize gene expression
X = normalize(X, norm='l1') #for positive non-zero entries, it's equivalent to: X /= X.sum(1).reshape(-1, 1)

#Save the node emmbeddings
np.savetxt("{}/data/output/{}/embedding/{}/embeddings.txt".format(project_path, FLAGS.dataset, FLAGS.embedding_method), X, delimiter="\t")
print("Embeddings saved in /data/output/{}/embedding/{}/embeddings.txt".format(FLAGS.dataset, FLAGS.embedding_method))