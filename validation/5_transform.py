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
flags.DEFINE_string('training_data', 'training_data.txt', 'Name of training data file.')
flags.DEFINE_string('testing_data', 'testing_data.txt', 'Name of testing data file.')
flags.DEFINE_string('validation_data', 'validation_data.txt', 'Name of validation data file.')

flags.DEFINE_string('clustering_method', 'geometric_ap', 'Name of output folder for the ustering method.')
flags.DEFINE_integer('distance_threshold', 3, 'Number of hops for node communication.')
flags.DEFINE_string('clusters', 'clusters.txt', 'Name of clusters file.')
flags.DEFINE_string('clusters_scores', 'clusters.scores.txt', 'Name of file of cluster scores.')
flags.DEFINE_list('labels', ["free", "metastasis"], 'List of class labels.')

flags.DEFINE_integer('hidden_layers', 2, 'Number of hidden layers.')
flags.DEFINE_integer('embedding_size', 16, 'Dimension of latent space.')
flags.DEFINE_float('p_val', 0.1, 'P-value for t-test.')

#Check data availability
if not os.path.isdir("{}/data/parsed_input/{}".format(project_path, FLAGS.dataset)):
    sys.exit("{} dataset is not available under data/parsed_input/".format(FLAGS.dataset))

if not os.path.isfile("{}/data/parsed_input/{}/p_val_{}/{}".format(project_path, FLAGS.dataset, FLAGS.p_val, FLAGS.training_data)):
    sys.exit("{} file is not available under /data/parsed_input/{}/p_val_{}/".format(FLAGS.training_data, FLAGS.dataset, FLAGS.p_val))
    
if not os.path.isfile("{}/data/parsed_input/{}/p_val_{}/{}".format(project_path, FLAGS.dataset, FLAGS.p_val, FLAGS.testing_data)):
    sys.exit("{} file is not available under /data/parsed_input/{}/p_val_{}/".format(FLAGS.testing_data, FLAGS.dataset, FLAGS.p_val))

if not os.path.isfile("{}/data/parsed_input/{}/p_val_{}/{}".format(project_path, FLAGS.dataset, FLAGS.p_val, FLAGS.validation_data)):
    sys.exit("{} file is not available under /data/parsed_input/{}/p_val_{}/".format(FLAGS.validation_data, FLAGS.dataset, FLAGS.p_val))
    
if not os.path.isfile("{}/data/output/{}/clustering/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold, FLAGS.clusters)):
    sys.exit("{} file is not available under /data/output/{}/clustering/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/".format(FLAGS.clusters, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold))
    
if not os.path.isfile("{}/data/output/{}/scoring/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold, FLAGS.clusters_scores)):
    sys.exit("{} file is not available under /data/output/{}/scoring/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/".format(FLAGS.clusters_scores, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold))

if not os.path.isdir("{}/data/output/{}/transformation/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold)):
    os.makedirs("{}/data/output/{}/transformation/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold))

print("----------------------------------------")
print("----------------------------------------")
print("Configuration:")
print("Dataset: {}".format(FLAGS.dataset))
print("Training data: {}".format(FLAGS.training_data))
print("Testing data: {}".format(FLAGS.testing_data))
print("Validation data: {}".format(FLAGS.validation_data))
print("Clustering method: {}".format(FLAGS.clustering_method))
print("Clustering radius: {}".format(FLAGS.distance_threshold))
print("Labels: {}".format(FLAGS.labels))
print("GCN hidden layers: {}".format(FLAGS.hidden_layers))
print("GCN embedding size: {}".format(FLAGS.embedding_size))
print("GCN p_val: {}".format(FLAGS.p_val))
print("----------------------------------------")
print("----------------------------------------")
    
#Use eps value to adjust transformed values with very low pdf
eps = np.finfo(np.double).eps
np.random.seed(seed=123)

def get_activity_data(clusters, data, labels_set):
    label_samples = data[:,-1]
    idx_label_samples = {}
    for label in labels_set:
        idx_label_samples[label] = np.ravel(np.where(label_samples==label))
    cluster_index = 0
    clusters.seek(0)
    nb_rows = len(clusters.readlines())
    nb_columns = data.shape[0]
    activity_scores = np.zeros((nb_rows, nb_columns), dtype=np.float32)
    clusters.seek(0)
    for line in clusters:
        line = line.strip()#To remove spaces
        members = np.asarray(line.split("\t")[1:], dtype=np.int32)
        current_cluster_features = np.asarray(data[:,members].T, dtype=np.float32) #Transpose to make the genes in row
        
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
            activity_scores[cluster_index, :] = np.concatenate((aggregated_phenotype1, aggregated_phenotype2))        
        cluster_index+=1
    activity_labels = np.concatenate(([labels_set[0]] * len(idx_label_samples[labels_set[0]]), [labels_set[1]] * len(idx_label_samples[labels_set[1]])))
    activity_data = np.zeros((nb_columns, nb_rows+1), dtype=object)
    activity_data[:, :-1] = activity_scores.T
    activity_data[:, -1] = activity_labels
    return activity_data
        
training_data = np.genfromtxt("{}/data/parsed_input/{}/p_val_{}/{}".format(project_path, FLAGS.dataset, FLAGS.p_val, FLAGS.training_data), dtype=np.dtype(str), skip_footer=1)
testing_data = np.genfromtxt("{}/data/parsed_input/{}/p_val_{}/{}".format(project_path, FLAGS.dataset, FLAGS.p_val, FLAGS.testing_data), dtype=np.dtype(str), skip_footer=1)
validation_data = np.genfromtxt("{}/data/parsed_input/{}/p_val_{}/{}".format(project_path, FLAGS.dataset, FLAGS.p_val, FLAGS.validation_data), dtype=np.dtype(str), skip_footer=1)
clusters = open("{}/data/output/{}/clustering/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold, FLAGS.clusters), encoding="utf-8")

print("Computing activity statistics of clusters...")
activity_training = get_activity_data(clusters, training_data, FLAGS.labels)
activity_testing = get_activity_data(clusters, testing_data, FLAGS.labels)
activity_validation = get_activity_data(clusters, validation_data, FLAGS.labels)
clusters.close()
np.savetxt("{}/data/output/{}/transformation/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/activity.training.txt".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold), activity_training, fmt="%s")
print("Successful generation of training activity scores in /data/output/{}/transformation/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/activity.training.txt".format(FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold))
np.savetxt("{}/data/output/{}/transformation/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/activity.testing.txt".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold), activity_testing, fmt="%s")
print("Successful generation of testing activity scores in /data/output/{}/transformation/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/activity.testing.txt".format(FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold))
np.savetxt("{}/data/output/{}/transformation/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/activity.validation.txt".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold), activity_validation, fmt="%s")
print("Successful generation of validation activity scores in /data/output/{}/transformation/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/activity.validation.txt".format(FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold))