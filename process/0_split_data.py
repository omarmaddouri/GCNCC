from __future__ import division
from __future__ import print_function

import tensorflow as tf
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind
import csv
import os
import sys
project_path = Path(__file__).resolve().parents[1]
sys.path.append(str(project_path))

# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'brc_microarray_usa', 'Dataset string.')
flags.DEFINE_float('p_value_threshold', 0.05, 'P-value for t-test.')
flags.DEFINE_float('test_split_size', 0.2, 'Size of test split.')
flags.DEFINE_float('val_split_size', 0.2, 'Size of validation split.')
flags.DEFINE_list('labels', ["free", "metastasis"], 'List of class labels.')

print("--------------------------------------------")
print("--------------------------------------------")
print("Hyper-parameters:")
print("P value threshold: {}".format(FLAGS.p_value_threshold))
print("val split size: {}".format(FLAGS.val_split_size))
print("test split size: {}".format(FLAGS.test_split_size))
print("Sample labels: {}".format(FLAGS.labels))
print("--------------------------------------------")
print("--------------------------------------------")
# Set random seed
seed = 123

#Check data availability
if not os.path.isdir("{}/data/parsed_input/{}".format(project_path, FLAGS.dataset)):
    os.makedirs("{}/data/parsed_input/{}".format(project_path, FLAGS.dataset))

if not os.path.isdir("{}/data/parsed_input/{}/p_val_{}".format(project_path, FLAGS.dataset, FLAGS.p_value_threshold)):
    os.makedirs("{}/data/parsed_input/{}/p_val_{}".format(project_path, FLAGS.dataset, FLAGS.p_value_threshold))
        
if not os.path.isfile("{}/data/parsed_input/{}/feature_label.txt".format(project_path, FLAGS.dataset)):
    sys.exit("{} feature_label data is not available under data/parsed_input/{}".format(FLAGS.dataset, FLAGS.dataset))
    
print("Split dataset into training ({}), validation ({}), and testing ({})...".format(1-(FLAGS.test_split_size+FLAGS.val_split_size), FLAGS.val_split_size, FLAGS.test_split_size))
def split_data(path="{}/data/parsed_input/".format(project_path), dataset=FLAGS.dataset):
    """Load dataset"""
    print('Loading {} dataset...'.format(dataset))

    features_labels = np.genfromtxt("{}{}/feature_label.txt".format(path, dataset), dtype=np.dtype(str))
    features = np.asarray(features_labels[:, :-1], dtype=np.float32)
    labels = features_labels[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=FLAGS.test_split_size, stratify=labels, random_state=seed)
    prorated_val_size = FLAGS.val_split_size / (1-FLAGS.test_split_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=prorated_val_size, stratify=y_train, random_state=seed)
    
    
    train = np.zeros((X_train.shape[0], X_train.shape[1]+1), dtype=object)
    test = np.zeros((X_test.shape[0], X_test.shape[1]+1), dtype=object)
    val = np.zeros((X_val.shape[0], X_val.shape[1]+1), dtype=object)
    train[:,:-1] = X_train
    train[:,-1] = y_train
    test[:,:-1] = X_test
    test[:,-1] = y_test
    val[:,:-1] = X_val
    val[:,-1] = y_val
    print("Size full dataset: {}".format(features.shape[0]))
    print("Size training dataset: {}".format(X_train.shape[0]))
    print("Size validation dataset: {}".format(X_val.shape[0]))
    print("Size testing dataset: {}".format(X_test.shape[0]))
    print("--------------------------------------------")
    return train, val, test

def label_columns(data):
    row_labels = data[:,-1]
    column_labels = []
    missing_ge_idx = []
    idx_class_samples = {}
    for label in FLAGS.labels:
        idx_class_samples[label] = np.where(row_labels==label)
    for c in range(data.shape[1]-1): #Last column contains labels
        if(np.count_nonzero(data[:,c]) > 0):
            phenotype1 = data[idx_class_samples[FLAGS.labels[0]],c].flatten()
            phenotype2 = data[idx_class_samples[FLAGS.labels[1]],c].flatten()
            _, p_value = ttest_ind(phenotype1, phenotype2, equal_var = False)
            l = "diff_express" if p_value < FLAGS.p_value_threshold else "non_diff_express"
            column_labels.append(l)
        else:
            l = "non_diff_express"
            column_labels.append(l)
            missing_ge_idx.append(c)
    column_labels.append("Label") #Last column reserved for labels
    print("T-test with a p-value threshold of: {}".format(FLAGS.p_value_threshold))
    print("Nb of genes differentially expressed: {}".format(column_labels.count("diff_express")))
    
    print("Nb of genes NON-differentially expressed: {}\n\
    (Note: {} genes have been labeled as NON-differentially expressed because of missing gene expression values)"\
    .format(column_labels.count("non_diff_express"), len(missing_ge_idx)))
    
    print("--------------------------------------------")
    return column_labels, missing_ge_idx

train, val, test = split_data()

print("Generate train data...")
training_data = np.zeros((train.shape[0]+1, train.shape[1]), dtype=object)
training_data[:-1,:] = train
training_data[-1,:], _ = label_columns(train)

print("Generate validation data...")
validation_data = np.zeros((val.shape[0]+1, val.shape[1]), dtype=object)
validation_data[:-1,:] = val
validation_data[-1,:], _ = label_columns(val)

print("Generate test data...")
testing_data = np.zeros((test.shape[0]+1, test.shape[1]), dtype=object)
testing_data[:-1,:] = test
testing_data[-1,:], _ = label_columns(test)

print("Generate full data...")
features_labels = np.genfromtxt("{}/data/parsed_input/{}/feature_label.txt".format(project_path, FLAGS.dataset), dtype=np.dtype(str))
full_data = np.zeros((features_labels.shape[0]+1, features_labels.shape[1]), dtype=object)
full_data[:-1, :-1] = np.asarray(features_labels[:, :-1], dtype=np.float32)
full_data[:-1,-1] = features_labels[:, -1]
full_data[-1,:], missing_ge_idx = label_columns(full_data[:-1,:])

np.savetxt("{}/data/parsed_input/{}/training_data.txt".format(project_path, FLAGS.dataset), training_data, fmt="%s")
np.savetxt("{}/data/parsed_input/{}/validation_data.txt".format(project_path, FLAGS.dataset), validation_data, fmt="%s")                                                                                                                                                          
np.savetxt("{}/data/parsed_input/{}/testing_data.txt".format(project_path, FLAGS.dataset), testing_data, fmt="%s")
np.savetxt("{}/data/parsed_input/{}/extended_testing_data.txt".format(project_path, FLAGS.dataset), np.vstack((training_data,testing_data)), fmt="%s")
np.savetxt("{}/data/parsed_input/{}/full_data.txt".format(project_path, FLAGS.dataset), full_data, fmt="%s")
np.savetxt("{}/data/parsed_input/{}/missing_ge_idx.txt".format(project_path, FLAGS.dataset), missing_ge_idx, fmt="%d")

print("Successful split of the dataset under data/parsed_input/{}".format(FLAGS.dataset))