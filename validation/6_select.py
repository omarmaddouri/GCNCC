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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.evaluate import PredefinedHoldoutSplit

# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'brc_microarray_usa', 'Dataset string.')
flags.DEFINE_string('classifier', 'logistic', 'Classifier used for feature selection.')
flags.DEFINE_string('training_activity', 'activity.training.txt', 'Name of training activity file.')
flags.DEFINE_string('validation_activity', 'activity.validation.txt', 'Name of training activity file.')
flags.DEFINE_string('testing_activity', 'activity.testing.txt', 'Name of testing activity file.')
flags.DEFINE_string('clustering_method', 'geometric_ap', 'Name of output folder for the ustering method.')
flags.DEFINE_integer('distance_threshold', 2, 'Number of hops for node communication.')
flags.DEFINE_string('clusters', 'clusters.txt', 'Name of clusters file.')
flags.DEFINE_string('clusters_symbols', 'clusters.symbols.txt', 'Name of clusters file with symbols.')
flags.DEFINE_string('clusters_scores', 'clusters.scores.txt', 'Name of file of cluster scores.')
flags.DEFINE_list('labels', ["free", "metastasis"], 'List of class labels.')
flags.DEFINE_integer('nb_features', 4, 'Number of features to be selected.')

flags.DEFINE_integer('hidden_layers', 3, 'Number of hidden layers.')
flags.DEFINE_integer('embedding_size', 64, 'Dimension of latent space.')
flags.DEFINE_float('p_val', 0.1, 'P-value for t-test.')

print("----------------------------------------")
print("----------------------------------------")
print("Configuration:")
print("Dataset: {}".format(FLAGS.dataset))
print("Classifier: {}".format(FLAGS.classifier))
print("Clustering Method: {}".format(FLAGS.clustering_method))
print("Clustering radius: {}".format(FLAGS.distance_threshold))
print("Class labels: {}".format(FLAGS.labels))
print("GCN hidden layers: {}".format(FLAGS.hidden_layers))
print("GCN embedding size: {}".format(FLAGS.embedding_size))
print("GCN p_val: {}".format(FLAGS.p_val))
print("----------------------------------------")
print("----------------------------------------")
#Check data availability
if not os.path.isdir("{}/data/output/{}".format(project_path, FLAGS.dataset)):
    sys.exit("{} dataset is not available under data/output/".format(FLAGS.dataset))

if not os.path.isfile("{}/data/output/{}/transformation/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold, FLAGS.training_activity)):
    sys.exit("{} file is not available under /data/output/{}/transformation/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/".format(FLAGS.training_activity, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold))

if not os.path.isfile("{}/data/output/{}/transformation/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold, FLAGS.validation_activity)):
    sys.exit("{} file is not available under /data/output/{}/transformation/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/".format(FLAGS.validation_activity, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold))
    
if not os.path.isfile("{}/data/output/{}/transformation/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold, FLAGS.testing_activity)):
    sys.exit("{} file is not available under /data/output/{}/transformation/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/".format(FLAGS.testing_activity, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold))
    
if not os.path.isfile("{}/data/output/{}/clustering/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold, FLAGS.clusters)):
    sys.exit("{} file is not available under /data/output/{}/clustering/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/".format(FLAGS.clusters, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold))
    
if not os.path.isfile("{}/data/output/{}/clustering/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold, FLAGS.clusters_symbols)):
    sys.exit("{} file is not available under /data/output/{}/clustering/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/".format(FLAGS.clusters_symbols, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold))
    
if not os.path.isfile("{}/data/output/{}/scoring/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold, FLAGS.clusters_scores)):
    sys.exit("{} file is not available under /data/output/{}/scoring/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/".format(FLAGS.clusters_scores, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold))

if not os.path.isdir("{}/data/output/{}/selection/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold)):
    os.makedirs("{}/data/output/{}/selection/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold))
    
cluster_scores = np.genfromtxt("{}/data/output/{}/scoring/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold, FLAGS.clusters_scores), dtype=object, skip_header=1)

nb_clusters = cluster_scores.shape[0]

cluster_scores[:,0] = np.asarray(cluster_scores[:,0], dtype=np.int32)
cluster_scores[:,1] = np.asarray(cluster_scores[:,1], dtype=np.float32)
sorted_cluster_scores = cluster_scores[cluster_scores[:,1].argsort()][::-1] #ascending sort then revert the order using [::-1]

reorder_column_indices = np.asarray(sorted_cluster_scores[:,0], dtype=np.int32)

training_activity = np.genfromtxt("{}/data/output/{}/transformation/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold, FLAGS.training_activity), dtype=np.dtype(str))
training_activity_features = np.asarray(training_activity[:,:-1], dtype=np.float32)
training_activity_labels = training_activity[:,-1]
#Reorder the features based on the cluster scores
training_activity_features = training_activity_features[:, reorder_column_indices]

validation_activity = np.genfromtxt("{}/data/output/{}/transformation/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold, FLAGS.validation_activity), dtype=np.dtype(str))
validation_activity_features = np.asarray(validation_activity[:,:-1], dtype=np.float32)
validation_activity_labels = validation_activity[:,-1]
#Reorder the features based on the cluster scores
validation_activity_features = validation_activity_features[:, reorder_column_indices]

testing_activity = np.genfromtxt("{}/data/output/{}/transformation/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold, FLAGS.testing_activity), dtype=np.dtype(str))
testing_activity_features = np.asarray(testing_activity[:,:-1], dtype=np.float32)
testing_activity_labels = testing_activity[:,-1]
#Reorder the features based on the cluster scores
testing_activity_features = testing_activity_features[:, reorder_column_indices]

if (FLAGS.classifier == "logistic"):
    selection_classifier = LogisticRegression(max_iter=1000)
elif (FLAGS.classifier == "svm_poly"):
    selection_classifier = SVC(kernel='poly')
if (FLAGS.classifier == "svm_rbf"):
    selection_classifier = SVC(kernel='rbf')

if(training_activity_features.shape[1] < FLAGS.nb_features):
    sfs_nb_features = training_activity_features.shape[1]
    print("Warning: the number of available clusters is less than the desired number of features to be selected. All available cluster features will be selected.")
else:
    sfs_nb_features = FLAGS.nb_features
    
print("SFS feature selection of best {} features...".format(sfs_nb_features))
test_val_indices = np.arange(training_activity_features.shape[0], training_activity_features.shape[0]+validation_activity_features.shape[0])

test_val = PredefinedHoldoutSplit(test_val_indices)
sfs_selector = SFS(selection_classifier, 
           k_features=sfs_nb_features, 
           forward=True, 
           floating=False, 
           verbose=2,
           scoring='roc_auc',
           cv=test_val)
X = np.concatenate([training_activity_features, validation_activity_features], axis=0)
y = np.concatenate([training_activity_labels,validation_activity_labels])
sfs_selector = sfs_selector.fit(X, y)

selection_classifier.fit(training_activity_features[:,list(sfs_selector.k_feature_idx_)], training_activity_labels)
test_auc = roc_auc_score(testing_activity_labels, selection_classifier.decision_function(testing_activity_features[:,list(sfs_selector.k_feature_idx_)]))

selected_clusters = reorder_column_indices[list(sfs_selector.k_feature_idx_)]
stats = np.zeros((8+sfs_nb_features, 2), dtype=object)
idx = 0

stats[idx,0] = "Validation AUC: "
stats[idx,1] = sfs_selector.k_score_
idx+=1

stats[idx,0] = "Testing AUC: "
stats[idx,1] = test_auc
idx+=1

disc_select_scores = []
for i in selected_clusters:
    disc_select_scores.append(cluster_scores[i,1])
stats[idx,0] = "Average discriminative power: "
stats[idx,1] = np.mean(disc_select_scores)
idx+=1

stats[idx,0] = "Selected clusters: "
stats[idx,1] = selected_clusters
idx+=1

symbols = {}
size_clusters = []
for f in range(sfs_nb_features):
    symbols[f] = np.genfromtxt("{}/data/output/{}/clustering/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold, FLAGS.clusters_symbols), delimiter="\t", dtype=np.dtype(str), skip_header = int(selected_clusters[f]), skip_footer = nb_clusters-(int(selected_clusters[f])+1)).flatten()
    size_clusters.append(len(symbols[f]))
stats[idx,0] = "Size clusters: "
stats[idx,1] = "{}:{}".format(np.sum(size_clusters), size_clusters)
idx+=1

cum_auc = []
for i in range(1,sfs_nb_features+1):
    cum_auc.append(sfs_selector.subsets_[i]['cv_scores'])
stats[idx,0] = "Cumulative validation AUC: "
stats[idx,1] = np.concatenate(cum_auc).ravel().tolist()
idx+=1

stats[idx,0] = "Discriminative power scores: "
stats[idx,1] = disc_select_scores
idx+=1

stats[idx,0] = "Gene symbols:"
stats[idx,1] = ""
idx+=1

for key, val in symbols.items():
    stats[idx,0] = ""
    stats[idx,1] = "\t".join(val)
    idx+=1

np.savetxt("{}/data/output/{}/selection/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/stats.{}.txt".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold, FLAGS.classifier), stats, fmt="%s")    
print("\n Results saved in /data/output/{}/selection/{}/{}L_dim_hidden_{}_p_val_{}_radius_{}/stats.{}.txt".format(FLAGS.dataset, FLAGS.clustering_method, FLAGS.hidden_layers, FLAGS.embedding_size, FLAGS.p_val, FLAGS.distance_threshold, FLAGS.classifier))    
#===============================================================================
# selection = [0]
# print("Forward Feature Selection: feature {}/{} retained".format(0, nb_clusters))
# baseline_accuracy = 0
# min_acc = 0.01 #minimum accuracy improvement to consider new cluster (1%)
# for i in range(1, nb_clusters):
#     logistic_classifier.fit(training_activity_features[:, selection], training_activity_labels)
#     preds = logistic_classifier.predict(testing_activity_features[:, selection])
#     acc = accuracy_score(testing_activity_labels, preds)
#     if( acc < baseline_accuracy + min_acc ):
#         print("Forward Feature Selection: feature {}/{} checked and rejected".format(i+1, nb_clusters))
#     else:
#         baseline_accuracy = acc
#         selection.append(i)
#         print("Forward Feature Selection: feature {}/{} checked and retained".format(i+1, nb_clusters))
# 
# print("The set of clusters to be used in classification is \n {}".format(reorder_column_indices[selection]))
#===============================================================================