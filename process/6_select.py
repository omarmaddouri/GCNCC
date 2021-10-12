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
flags.DEFINE_string('train_dataset', 'brc_microarray_usa', 'Train dataset string.')
flags.DEFINE_string('test_dataset', 'brc_microarray_usa', 'Test dataset string.')
flags.DEFINE_string('classifier', 'logistic', 'Classifier used for feature selection.')
flags.DEFINE_string('training_activity', 'activity.training.txt', 'Name of training activity file.')
flags.DEFINE_string('validation_activity', 'activity.validation.txt', 'Name of training activity file.')                                                                                                        
flags.DEFINE_string('testing_activity', 'activity.testing.txt', 'Name of testing activity file.')

flags.DEFINE_string('clustering_method', 'geometric_ap', 'Name of output folder for the clustering method.')
flags.DEFINE_string('embedding_method', 'gcn', 'Name of the embedding method.')

flags.DEFINE_string('clusters', 'clusters.txt', 'Name of clusters file.')
flags.DEFINE_string('clusters_symbols', 'clusters.symbols.txt', 'Name of clusters file with symbols.')
flags.DEFINE_string('clusters_scores', 'clusters.scores.txt', 'Name of file of cluster scores.')
flags.DEFINE_list('labels', ["free", "metastasis"], 'List of class labels.')
flags.DEFINE_integer('nb_features', 4, 'Number of features to be selected.')

#Check data availability
if not os.path.isdir("{}/data/output/{}".format(project_path, FLAGS.train_dataset)):
    sys.exit("{} train dataset is not available under data/output/".format(FLAGS.train_dataset))
    
if not os.path.isdir("{}/data/output/{}".format(project_path, FLAGS.test_dataset)):
    sys.exit("{} test dataset is not available under data/output/".format(FLAGS.test_dataset))
    
if not os.path.isfile("{}/data/output/{}/transformation/{}/{}/{}".format(project_path, FLAGS.train_dataset, FLAGS.clustering_method, FLAGS.embedding_method, FLAGS.training_activity)):
    sys.exit("{} file is not available under /data/output/{}/transformation/{}/{}/".format(FLAGS.training_activity, FLAGS.train_dataset, FLAGS.clustering_method, FLAGS.embedding_method))
    
if not os.path.isfile("{}/data/output/{}/transformation/{}/{}/{}".format(project_path, FLAGS.test_dataset, FLAGS.clustering_method, FLAGS.embedding_method, FLAGS.testing_activity)):
    sys.exit("{} file is not available under /data/output/{}/transformation/{}/{}/".format(FLAGS.testing_activity, FLAGS.test_dataset, FLAGS.clustering_method, FLAGS.embedding_method))
    
if not os.path.isfile("{}/data/output/{}/clustering/{}/{}/{}".format(project_path, FLAGS.train_dataset, FLAGS.clustering_method, FLAGS.embedding_method, FLAGS.clusters)):
    sys.exit("{} file is not available under /data/output/{}/clustering/{}/{}/".format(FLAGS.clusters, FLAGS.train_dataset, FLAGS.clustering_method, FLAGS.embedding_method))
    
if not os.path.isfile("{}/data/output/{}/clustering/{}/{}/{}".format(project_path, FLAGS.train_dataset, FLAGS.clustering_method, FLAGS.embedding_method, FLAGS.clusters_symbols)):
    sys.exit("{} file is not available under /data/output/{}/clustering/{}/{}/".format(FLAGS.clusters_symbols, FLAGS.train_dataset, FLAGS.clustering_method, FLAGS.embedding_method))
    
if not os.path.isfile("{}/data/output/{}/scoring/{}/{}/{}".format(project_path, FLAGS.train_dataset, FLAGS.clustering_method, FLAGS.embedding_method, FLAGS.clusters_scores)):
    sys.exit("{} file is not available under /data/output/{}/scoring/{}/{}/".format(FLAGS.clusters_scores, FLAGS.train_dataset, FLAGS.clustering_method, FLAGS.embedding_method))

if not os.path.isdir("{}/data/output/{}/selection/{}/{}/{}_vs_{}".format(project_path, FLAGS.train_dataset, FLAGS.clustering_method, FLAGS.embedding_method, FLAGS.train_dataset, FLAGS.test_dataset)):
    os.makedirs("{}/data/output/{}/selection/{}/{}/{}_vs_{}".format(project_path, FLAGS.train_dataset, FLAGS.clustering_method, FLAGS.embedding_method, FLAGS.train_dataset, FLAGS.test_dataset))

print("----------------------------------------")
print("----------------------------------------")
print("Configuration:")
print("Train dataset: {}".format(FLAGS.train_dataset))
print("Test dataset: {}".format(FLAGS.test_dataset))
print("Clustering Method: {}".format(FLAGS.clustering_method))
print("Embedding Method: {}".format(FLAGS.embedding_method))
print("Class labels: {}".format(FLAGS.labels))
print("----------------------------------------")
print("----------------------------------------")

if(FLAGS.train_dataset == FLAGS.test_dataset):    
    cluster_scores = np.genfromtxt("{}/data/output/{}/scoring/{}/{}/{}".format(project_path, FLAGS.train_dataset, FLAGS.clustering_method, FLAGS.embedding_method, FLAGS.clusters_scores), dtype=object, skip_header=1)
else:
    cluster_scores = np.genfromtxt("{}/data/output/{}/scoring/{}/{}/cross.{}.{}".format(project_path, FLAGS.train_dataset, FLAGS.clustering_method, FLAGS.embedding_method, FLAGS.test_dataset, FLAGS.clusters_scores), dtype=object, skip_header=1)

nb_clusters = cluster_scores.shape[0]

cluster_scores[:,0] = np.asarray(cluster_scores[:,0], dtype=np.int32)
cluster_scores[:,1] = np.asarray(cluster_scores[:,1], dtype=np.float32)

cluster_scores[:,2] = np.asarray(cluster_scores[:,2], dtype=np.int32)
filtered_cluster_scores = cluster_scores[cluster_scores[:,2] <= 250]
filtered_cluster_scores = filtered_cluster_scores[filtered_cluster_scores[:,2] >= 10]
if(filtered_cluster_scores.shape[0] < FLAGS.nb_features):
    filtered_cluster_scores = cluster_scores

sorted_cluster_scores = filtered_cluster_scores[filtered_cluster_scores[:,1].argsort()][::-1] #ascending sort then revert the order using [::-1]

reorder_column_indices = np.asarray(sorted_cluster_scores[:,0], dtype=np.int32)

training_activity = np.genfromtxt("{}/data/output/{}/transformation/{}/{}/{}".format(project_path, FLAGS.train_dataset, FLAGS.clustering_method, FLAGS.embedding_method, FLAGS.training_activity), dtype=np.dtype(str))
training_activity_features = np.asarray(training_activity[:,:-1], dtype=np.float32)
training_activity_labels = training_activity[:,-1]
#Reorder the features based on the cluster scores
training_activity_features = training_activity_features[:, reorder_column_indices]

if(FLAGS.train_dataset == FLAGS.test_dataset):
    validation_activity = np.genfromtxt("{}/data/output/{}/transformation/{}/{}/{}".format(project_path, FLAGS.train_dataset, FLAGS.clustering_method, FLAGS.embedding_method, FLAGS.validation_activity), dtype=np.dtype(str))
else:
    validation_activity = np.genfromtxt("{}/data/output/{}/transformation/{}/{}/cross.{}.{}".format(project_path, FLAGS.train_dataset, FLAGS.clustering_method, FLAGS.embedding_method, FLAGS.test_dataset, FLAGS.validation_activity), dtype=np.dtype(str))
    
validation_activity_features = np.asarray(validation_activity[:,:-1], dtype=np.float32)
validation_activity_labels = validation_activity[:,-1]
#Reorder the features based on the cluster scores
validation_activity_features = validation_activity_features[:, reorder_column_indices]

if(FLAGS.train_dataset == FLAGS.test_dataset):
    testing_activity = np.genfromtxt("{}/data/output/{}/transformation/{}/{}/{}".format(project_path, FLAGS.train_dataset, FLAGS.clustering_method, FLAGS.embedding_method, FLAGS.testing_activity), dtype=np.dtype(str))
else:
    testing_activity = np.genfromtxt("{}/data/output/{}/transformation/{}/{}/cross.{}.{}".format(project_path, FLAGS.train_dataset, FLAGS.clustering_method, FLAGS.embedding_method, FLAGS.test_dataset, FLAGS.testing_activity), dtype=np.dtype(str))
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
           fixed_features=(0,),
           cv=test_val,
           n_jobs=8)#-1)
X = np.concatenate([training_activity_features, validation_activity_features], axis=0)
y = np.concatenate([training_activity_labels,validation_activity_labels])
sfs_selector = sfs_selector.fit(X, y)

selection_classifier.fit(training_activity_features[:,list(sfs_selector.k_feature_idx_)], training_activity_labels)
test_auc = roc_auc_score(testing_activity_labels, selection_classifier.decision_function(testing_activity_features[:,list(sfs_selector.k_feature_idx_)]))
print("\n Performance on the testing set: {}\n".format(test_auc))
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
    symbols[f] = np.genfromtxt("{}/data/output/{}/clustering/{}/{}/{}".format(project_path, FLAGS.train_dataset, FLAGS.clustering_method, FLAGS.embedding_method, FLAGS.clusters_symbols), delimiter="\t", dtype=np.dtype(str), skip_header = int(selected_clusters[f]), skip_footer = nb_clusters-(int(selected_clusters[f])+1)).flatten()
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

np.savetxt("{}/data/output/{}/selection/{}/{}/{}_vs_{}/stats.txt".format(project_path, FLAGS.train_dataset, FLAGS.clustering_method, FLAGS.embedding_method, FLAGS.train_dataset, FLAGS.test_dataset), stats, fmt="%s")    
print("\n Results saved in /data/output/{}/selection/{}/{}/{}_vs_{}/stats.txt".format(FLAGS.train_dataset, FLAGS.clustering_method, FLAGS.embedding_method, FLAGS.train_dataset, FLAGS.test_dataset))    
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