from __future__ import division
from __future__ import print_function

from pathlib import Path
from random import random
import sys
project_path = Path(__file__).resolve().parents[1]
sys.path.append(str(project_path))

import tensorflow as tf
import os
import scipy.sparse as sp
import numpy as np
from core.GeometricAffinityPropagation import AffinityPropagation as GeometricAffinityPropagation
from sklearn.metrics import euclidean_distances
import mygene
from collections import defaultdict
import csv

# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'brc_microarray_usa', 'Dataset string.')
flags.DEFINE_string('embeddings', 'embeddings.txt', 'Name of file that contains the latent node features.')
flags.DEFINE_string('embedding_method', 'gcn', 'Name of output folder for the embedding method.')

flags.DEFINE_integer('max_iter', 300, 'Maximum number of iterations.')
flags.DEFINE_integer('convergence_iter', 15, 'Number of iterations with no change in the number of estimated clusters that stops the convergence.')
flags.DEFINE_float('damp_factor', 0.9, 'Damping factor for AP message updates.')

flags.DEFINE_integer('distance_threshold', 2, 'Neighborhood threshold.')
flags.DEFINE_string('similarity_metric', 'shortest_path', 'Network similarity metric.')
flags.DEFINE_bool('save_mask', True, 'Whether to save the clustering mask or not.')
flags.DEFINE_string('saving_path', '{}/data/output/mask/'.format(project_path), 'Path for saving the mask.')

flags.DEFINE_float('quantile', 0.95, 'Quantile used to determine the preference value.')
flags.DEFINE_integer('desired_nb_clusters', 500, 'Desired number of clusters.')
flags.DEFINE_integer('tolerance_nb_clusters', 100, 'Tolerance level for the number of obtained clusters.')

flags.DEFINE_string('clustering_method', 'geometric_ap', 'Name of output folder for the clustering method.')
flags.DEFINE_string('clusters_output', 'clusters.txt', 'Name of desired output file of obtained clusters.')
flags.DEFINE_string('modularity', 'modularity.txt', 'Name of output file of resulting modularity metric.')

flags.DEFINE_integer('hidden_layers', 3, 'Number of hidden layers.')
flags.DEFINE_integer('embedding_size', 64, 'Dimension of latent space.')
flags.DEFINE_float('p_val', 0.1, 'P-value for t-test.')

#Check data availability
if not os.path.isfile("{}/data/output/network/ppi.adjacency.npz".format(project_path)):
    sys.exit("Network adjacency file is not available under /data/output/network/")
    
if not os.path.isfile("{}/data/output/{}/embedding/{}/{}L_dim_hidden_{}_p_val_{}/{}".format(project_path, FLAGS.dataset, FLAGS.embedding_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.embeddings)):
    sys.exit("{} file is not available under /data/output/{}/embedding/{}/{}L_dim_hidden_{}_p_val_{}/".format(FLAGS.embeddings, FLAGS.dataset, FLAGS.embedding_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val))
    
if not os.path.isdir("{}/data/output/{}/clustering/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold)):
    os.makedirs("{}/data/output/{}/clustering/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold))
    
if not os.path.isdir("{}/data/output/mask/{}".format(project_path, FLAGS.dataset)):
    os.makedirs("{}/data/output/mask/{}".format(project_path, FLAGS.dataset))    

print("----------------------------------------")
print("----------------------------------------")
print("Clustering configuration:")
print("Dataset: {}".format(FLAGS.dataset))
print("Embedding method: {}".format(FLAGS.embedding_method))
print("Clustering Method: {}".format(FLAGS.clustering_method))
print("Clustering radius: {}".format(FLAGS.distance_threshold))
print("GCN hidden layers: {}".format(FLAGS.hidden_layers))
print("GCN embedding size: {}".format(FLAGS.embedding_size))
print("GCN p_val: {}".format(FLAGS.p_val))
print("----------------------------------------")
print("----------------------------------------")
adj = sp.load_npz("{}/data/output/network/ppi.adjacency.npz".format(project_path))
adj = adj.toarray()
distance_threshold = FLAGS.distance_threshold
similarity_metric = FLAGS.similarity_metric
max_iter = FLAGS.max_iter
convergence_iter = FLAGS.convergence_iter
damp_factor = FLAGS.damp_factor

def save_clusters(labels, cluster_centers_indices, modularity):
    clusters = defaultdict(list)
    for i in range(len(labels)):
        clusters["Exemplar_{}".format(cluster_centers_indices[labels[i]])].append(i)
    with open("{}/data/output/{}/clustering/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold, FLAGS.clusters_output), "w", newline='', encoding="utf-8") as f:
        w_clusters = csv.writer(f, delimiter ='\t')
        for key, val in clusters.items():
            line = []
            line.append(key)
            for item in val:
                line.append(item)
            w_clusters.writerow(line)
    
    modularity_dict = {}
    modularity_dict["Modularity metric:"] = modularity
    with open("{}/data/output/{}/clustering/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold, FLAGS.modularity), "w", newline='', encoding="utf-8") as f:
        w_modularity = csv.writer(f, delimiter ='\t')
        for key, val in modularity_dict.items():
            w_modularity.writerow([key, val])
#===============================================================================
# Read the node embeddings                
#===============================================================================
embeddings = np.genfromtxt("{}/data/output/{}/embedding/{}/{}L_dim_hidden_{}_p_val_{}/{}".format(project_path, FLAGS.dataset, FLAGS.embedding_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.embeddings), dtype=np.float32)

#===============================================================================
# Start the clustering by geometric affinity propagation
#===============================================================================
print("Clustering by Geometric Affinity Propagation ...")
print("Settings:")
print("Maximum iterations: {}".format(max_iter))
print("Convergence iterations: {}".format(convergence_iter))
print("Damping factor: {}".format(damp_factor))
print("Network neighborhood threshold: {}".format(distance_threshold))
print("------------------------------")

similarities = -euclidean_distances(embeddings, squared=True)

n_clusters = 0
wait_convergence = 5
Q = np.quantile(similarities, FLAGS.quantile)
if(Q==0):
    preference = np.median(similarities)
else:
    preference = np.median(similarities[np.where(similarities>Q)])
        
while( (np.absolute(FLAGS.desired_nb_clusters-n_clusters) > FLAGS.tolerance_nb_clusters) and wait_convergence>0):
    af = GeometricAffinityPropagation(damping=damp_factor, max_iter=max_iter, convergence_iter=convergence_iter, copy=False, preference=preference, 
                             affinity='euclidean', verbose=True, random_state=0, adjacency=adj, 
                             node_similarity_mode=similarity_metric, node_similarity_threshold=distance_threshold,
                             save_mask=FLAGS.save_mask, saving_path=FLAGS.saving_path, enable_label_smoothing=False, dataset=FLAGS.dataset).fit(embeddings)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    modularity = af.modularity_
    n_clusters = len(cluster_centers_indices)
    if(n_clusters > FLAGS.desired_nb_clusters):
        preference = (preference*2)-(np.finfo(np.float32).eps*random()*10)
    else:
        preference = (preference/2)-(np.finfo(np.float32).eps*random()*10)
    if(n_clusters>0):
        wait_convergence = wait_convergence-1
print("Number of obtained clusters: {}".format(n_clusters))
print("Clustering modularity: {}".format(modularity))
if n_clusters==0:
    sys.exit("Geometric AP did not converge: Increase max_iter for convergence or tune other hyperparameters (i.e: damping factor, convergence iteration threshold)")
 
print("Save clustering results ...")
save_clusters(labels, cluster_centers_indices, modularity)
print("Clustering results saved")
