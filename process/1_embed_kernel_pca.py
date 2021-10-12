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


# Set random seed
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)

# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'brc_microarray_usa', 'Dataset string.')
flags.DEFINE_string('embedding_method', 'kernel_pca', 'Name of the embedding method.')
flags.DEFINE_integer('embedding_size', 16, 'Dimension of latent space.')

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
print("Dimension of latent space: {}".format(FLAGS.embedding_size))
print("--------------------------------------------")
print("--------------------------------------------")

# Prepare Data
X, _, _ = load_training_data(dataset=FLAGS.dataset)
transformer = KernelPCA(n_components=FLAGS.embedding_size, kernel='linear')
embeddings = transformer.fit_transform(X)
#Save the node emmbeddings
np.savetxt("{}/data/output/{}/embedding/{}/embeddings.txt".format(project_path, FLAGS.dataset, FLAGS.embedding_method), embeddings, delimiter="\t")
print("Embeddings saved in /data/output/{}/embedding/{}/embeddings.txt".format(FLAGS.dataset, FLAGS.embedding_method))