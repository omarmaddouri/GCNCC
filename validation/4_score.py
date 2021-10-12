from __future__ import division
from __future__ import print_function

from pathlib import Path
import sys
project_path = Path(__file__).resolve().parents[1]
sys.path.append(str(project_path))

import tensorflow as tf
import os
import numpy as np
from collections import defaultdict
import csv
from scipy import stats

# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'brc_microarray_usa', 'Dataset string.')
flags.DEFINE_string('clustering_method', 'geometric_ap', 'Name of clustering method.')
flags.DEFINE_integer('distance_threshold', 3, 'Number of hops for node communication.')

flags.DEFINE_string('scoring_data', 'full_data.txt', 'Full data to be used for clusters scoring.')

flags.DEFINE_string('clusters', 'clusters.txt', 'Name of clusters file.')
flags.DEFINE_list('labels', ["free", "metastasis"], 'List of class labels.')

flags.DEFINE_integer('hidden_layers', 2, 'Number of hidden layers.')
flags.DEFINE_integer('embedding_size', 16, 'Dimension of latent space.')
flags.DEFINE_float('p_val', 0.1, 'P-value for t-test.')

#Check data availability
if not os.path.isdir("{}/data/parsed_input/{}".format(project_path, FLAGS.dataset)):
    sys.exit("{} dataset is not available under data/parsed_input/".format(FLAGS.dataset))

if not os.path.isfile("{}/data/parsed_input/{}/p_val_{}/{}".format(project_path, FLAGS.dataset, FLAGS.p_val, FLAGS.scoring_data)):
    sys.exit("{} file is not available under /data/parsed_input/{}/p_val_{}/".format(FLAGS.scoring_data, FLAGS.dataset, FLAGS.p_val))
    
if not os.path.isfile("{}/data/output/{}/clustering/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold, FLAGS.clusters)):
    sys.exit("{} file is not available under /data/output/{}/clustering/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}".format(FLAGS.clusters, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold))

if not os.path.isdir("{}/data/output/{}/scoring/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold)):
    os.makedirs("{}/data/output/{}/scoring/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold))

print("----------------------------------------")
print("----------------------------------------")
print("Configuration:")
print("Dataset: {}".format(FLAGS.dataset))
print("Clustering Method: {}".format(FLAGS.clustering_method))
print("Clustering radius: {}".format(FLAGS.distance_threshold))
print("Class labels: {}".format(FLAGS.labels))
print("GCN hidden layers: {}".format(FLAGS.hidden_layers))
print("GCN embedding size: {}".format(FLAGS.embedding_size))
print("GCN p_val: {}".format(FLAGS.p_val))
print("----------------------------------------")
print("----------------------------------------")
        
scoring_data = np.genfromtxt("{}/data/parsed_input/{}/p_val_{}/{}".format(project_path, FLAGS.dataset, FLAGS.p_val, FLAGS.scoring_data), dtype=np.dtype(str), skip_footer=1)
clusters = open("{}/data/output/{}/clustering/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold, FLAGS.clusters), encoding="utf-8")       
#Use eps value to adjust transformed values with very low pdf
eps = np.finfo(np.double).eps
np.random.seed(seed=123)

label_samples = scoring_data[:,-1]
idx_label_samples = {}
for label in FLAGS.labels:
    idx_label_samples[label] = np.ravel(np.where(label_samples==label))
score_clusters = {}
score_clusters["cluster_index"] = "activity_score"
cluster_index = 0
print("Scoring of clusters...")
for line in clusters:
    line = line.strip()#To remove spaces
    members = np.asarray(line.split("\t")[1:], dtype=np.int32)
    current_cluster_features = np.asarray(scoring_data[:,members].T, dtype=np.float32) #Transpose to make the genes in row
    
    # Remove missing gene expressions
    current_cluster_features = current_cluster_features[~np.all(current_cluster_features == 0, axis=1)]
    
    if(current_cluster_features.shape[0] != 0):
        phenotype1 = current_cluster_features[:, idx_label_samples[FLAGS.labels[0]]] #Select the samples under phenotype 1
        phenotype2 = current_cluster_features[:, idx_label_samples[FLAGS.labels[1]]] #Select the samples under phenotype 2
        mu1 = np.mean(phenotype1, axis=1)
        std1 = np.std(phenotype1, axis=1)
        std1[std1<eps] = eps
        
        mu2 = np.mean(phenotype2, axis=1)
        std2 = np.std(phenotype2, axis=1)
        std2[std2<eps] = eps
        
        l = phenotype1.shape[1]
        for i in range(current_cluster_features.shape[0]):
            row = np.concatenate((phenotype1[i,:], phenotype2[i,:]))
            N1 = stats.norm(mu1[i],std1[i]).pdf(row)
            N2 = stats.norm(mu2[i],std2[i]).pdf(row)
            #Cutoff of outliers
            N1[N1<eps] = eps
            N2[N2<eps] = eps
            row = np.log(N1) - np.log(N2)
            if(np.count_nonzero(row) == 0):
                transformed_row = row
            else:        
                transformed_row = stats.zscore(row)
            phenotype1[i,:] = transformed_row[:l]
            phenotype2[i,:] = transformed_row[l:]
        
        aggregated_phenotype1 = np.sum(phenotype1, axis=0, dtype=np.float32)
        aggregated_phenotype2 = np.sum(phenotype2, axis=0, dtype=np.float32)
    
        t, p = stats.ttest_ind(aggregated_phenotype1, aggregated_phenotype2, equal_var = False)
    else:
        t=0
        p=1
    score_clusters[cluster_index] = np.abs(t)
    cluster_index+=1
clusters.close()
with open("{}/data/output/{}/scoring/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/clusters.scores.txt".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold), "w", newline='', encoding="utf-8") as f:
    w_scores = csv.writer(f, delimiter ='\t')
    for key, val in score_clusters.items():
        w_scores.writerow([key, val])
print("Successful generation of cluster scores in /data/output/{}/scoring/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/clusters.scores.txt".format(FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold))        