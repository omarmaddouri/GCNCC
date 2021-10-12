from __future__ import division
from __future__ import print_function

from pathlib import Path
import sys
project_path = Path(__file__).resolve().parents[1]
sys.path.append(str(project_path))

import tensorflow as tf
import os
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import euclidean_distances
import mygene
from collections import defaultdict
import csv
import markov_clustering as mc
import networkx as nx
import operator

# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'brc_microarray_usa', 'Dataset string.')
flags.DEFINE_string('embeddings', 'full_data.txt', 'Name of file that contains the latent node features.')
flags.DEFINE_string('embedding_method', 'ge', 'Name of the embedding method.')

flags.DEFINE_string('clustering_method', 'wgcna', 'Name of the clustering method.')

#Check data availability
if not os.path.isfile("{}/data/parsed_input/{}/{}".format(project_path, FLAGS.dataset, FLAGS.embeddings)):
    sys.exit("{} file is not available under /data/parsed_input/{}/".format(FLAGS.embeddings, FLAGS.dataset))

if not os.path.isdir("{}/data/output/{}/clustering/{}/{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.embedding_method)):
    os.makedirs("{}/data/output/{}/clustering/{}/{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.embedding_method))

print("----------------------------------------")
print("----------------------------------------")
print("Clustering configuration:")
print("Dataset: {}".format(FLAGS.dataset))
print("Embedding method: {}".format(FLAGS.embedding_method))
print("Clustering Method: {}".format(FLAGS.clustering_method))
print("----------------------------------------")
print("----------------------------------------")

#===============================================================================
# Read the node embeddings                
#===============================================================================
embeddings = np.genfromtxt("{}/data/parsed_input/{}/{}".format(project_path, FLAGS.dataset, FLAGS.embeddings), skip_footer=1, dtype=np.float64)
wgcna_matrix = np.zeros((embeddings.shape[0]+1, embeddings.shape[1]), dtype=np.object)
for i in range(wgcna_matrix.shape[0]):
    wgcna_matrix[i,0] = i
for j in range(wgcna_matrix.shape[1]):
    wgcna_matrix[0,j] = j
wgcna_matrix[1:, 1:] = embeddings[:,:-1]

#Save wgcna matrix
np.savetxt("{}/data/output/{}/clustering/{}/{}/wgcna_matrix.txt".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.embedding_method), wgcna_matrix.T, fmt="%s", delimiter="\t")
print("WGCNA matrix saved in /data/output/{}/clustering/{}/{}/wgcna_matrix.txt".format(FLAGS.dataset, FLAGS.clustering_method, FLAGS.embedding_method))
