"""
Much of this implementation is inspired by:
https://github.com/tkipf/keras-gcn
And
https://github.com/vermaMachineLearning/keras-deep-graph-learning/blob/master/keras_dgl/layers/graph_cnn_layer.py
"""
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
flags.DEFINE_string('embedding_method', 'gcn', 'Name of output folder for the embedding method.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('embedding_size', 8, 'Dimension of latent space.')

flags.DEFINE_bool('save_filter', True, 'Save graph convolution filter to speed up computation.')
flags.DEFINE_string('saving_path', '{}/data/output/filter/'.format(project_path), 'Path for saving the mask.')

#flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('p_value_threshold', 0.05, 'P-value for t-test.')
flags.DEFINE_integer('early_stopping', 50, 'Tolerance for early stopping (# of epochs).')

#Check dataset availability
if not os.path.isdir("{}/data/parsed_input/{}".format(project_path, FLAGS.dataset)):
    sys.exit("{} dataset is not available under data/parsed_input/".format(FLAGS.dataset))
    
if not os.path.isdir("{}/data/output/{}/embedding/{}".format(project_path, FLAGS.dataset, FLAGS.embedding_method)):
    os.makedirs("{}/data/output/{}/embedding/{}".format(project_path, FLAGS.dataset, FLAGS.embedding_method))
    
if not os.path.isdir("{}/data/output/{}/embedding/{}/4L_dim_hidden_{}_p_val_{}".format(project_path, FLAGS.dataset, FLAGS.embedding_method, FLAGS.embedding_size, FLAGS.p_value_threshold)):
    os.makedirs("{}/data/output/{}/embedding/{}/4L_dim_hidden_{}_p_val_{}".format(project_path, FLAGS.dataset, FLAGS.embedding_method, FLAGS.embedding_size, FLAGS.p_value_threshold))
    
if not os.path.isdir("{}/data/output/filter".format(project_path)):
    os.makedirs("{}/data/output/filter".format(project_path))

print("--------------------------------------------")
print("--------------------------------------------")
print("Hyper-parameters:")
print("P value threshold: {}".format(FLAGS.p_value_threshold))
print("Number of epochs to train: {}".format(FLAGS.epochs))
print("Dimension of latent space: {}".format(FLAGS.embedding_size))
print("Tolerance for early stopping: {}".format(FLAGS.early_stopping))
print("--------------------------------------------")
print("--------------------------------------------")

# Prepare Data
X, A, Y = load_training_data_tuning(dataset=FLAGS.dataset, p_val=FLAGS.p_value_threshold)

Y_train, Y_val, Y_test, train_idx, val_idx, test_idx, train_mask = get_splits_for_training_tuning(Y, dataset=FLAGS.dataset, p_val=FLAGS.p_value_threshold)

# Normalize gene expression
X = normalize(X, norm='l1') #for positive non-zero entries, it's equivalent to: X /= X.sum(1).reshape(-1, 1)

# Build Graph Convolution filters
if not FLAGS.save_filter or not os.path.isfile("{}/filter.npy".format(FLAGS.saving_path)):
#     SYM_NORM = True
#     A_norm = preprocess_adj(A, SYM_NORM)
#     num_filters = 4
#     A_norm_2 = np.linalg.matrix_power(A_norm.todense(), 2).astype(np.float32)
#     A_norm_3 = np.linalg.matrix_power(A_norm.todense(), 3).astype(np.float32)
#     graph_conv_filters = sp.vstack([sp.eye(A_norm.shape[0]), A_norm, sp.csr_matrix(A_norm_2), sp.csr_matrix(A_norm_3)])
#     graph_conv_filters = K.constant(graph_conv_filters.todense())
    SYM_NORM = True
    A_norm = preprocess_adj(A, SYM_NORM)
    num_filters = 3
    A_norm_2 = np.linalg.matrix_power(A_norm.todense(), 2).astype(np.float32)
    graph_conv_filters = sp.vstack([sp.eye(A_norm.shape[0]), A_norm, sp.csr_matrix(A_norm_2)])
    graph_conv_filters = K.constant(graph_conv_filters.todense())
    if(FLAGS.save_filter):
        np.save("{}/filter".format(FLAGS.saving_path), graph_conv_filters)
else:
    num_filters = 3
    graph_conv_filters = np.load("{}/filter.npy".format(FLAGS.saving_path), allow_pickle=True)
    graph_conv_filters = K.constant(graph_conv_filters)

#Build Model
model = Sequential()
model.add(GraphCNN(X.shape[1], num_filters, graph_conv_filters, input_shape=(X.shape[1],), activation='tanh', kernel_regularizer=l2(5e-4)))
model.add(GraphCNN(FLAGS.embedding_size*8, num_filters, graph_conv_filters, input_shape=(X.shape[1],), activation='tanh', kernel_regularizer=l2(5e-4)))
model.add(GraphCNN(FLAGS.embedding_size*4, num_filters, graph_conv_filters, input_shape=(X.shape[1],), activation='tanh', kernel_regularizer=l2(5e-4)))
model.add(GraphCNN(FLAGS.embedding_size*2, num_filters, graph_conv_filters, input_shape=(X.shape[1],), activation='tanh', kernel_regularizer=l2(5e-4)))
model.add(GraphCNN(FLAGS.embedding_size, num_filters, graph_conv_filters, input_shape=(X.shape[1],), activation='tanh', kernel_regularizer=l2(5e-4)))
model.add(GraphCNN(Y.shape[1], num_filters, graph_conv_filters, activation='softmax', kernel_regularizer=l2(5e-4)))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['acc'])

# Helper variables for main training loop
wait = 0
preds = None
best_val_loss = 99999
PATIENCE = FLAGS.early_stopping  #Early stopping patience
nb_epochs = FLAGS.epochs

for epoch in range(nb_epochs):
    # Log wall-clock time
    t = time.time()
    
    model.fit(X, Y_train, sample_weight=train_mask, batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)
    Y_pred = model.predict(X, batch_size=A.shape[0])
    #Intercept features[Start]--------------------------------------------------
    model_intercept = Model(inputs=model.input, outputs=model.layers[4].output)
    embeddings = model_intercept.predict(X, batch_size=A.shape[0])
    #Intercept features[End]----------------------------------------------------
    train_val_test_loss, train_val_test_acc = evaluate_preds(Y_pred, [Y_train, Y_val, Y_test], [train_idx, val_idx, test_idx])
    
    print("Epoch: {:04d}---".format(epoch),\
          "train_loss= {:.4f}---".format(train_val_test_loss[0]),\
          "train_acc= {:.4f}---".format(train_val_test_acc[0]),\
          "val_loss= {:.4f}---".format(train_val_test_loss[1]),\
          "val_acc= {:.4f}---".format(train_val_test_acc[1]),\
          "test_loss= {:.4f}---".format(train_val_test_loss[2]),\
          "test_acc= {:.4f}".format(train_val_test_acc[2]),\
          "time= {:.4f}".format(time.time() - t))
    # Early stopping
    if train_val_test_loss[0] < best_val_loss:        
        best_train_loss = train_val_test_loss[0]
        best_train_acc = train_val_test_acc[0]
        
        best_val_loss = train_val_test_loss[1]
        best_val_acc = train_val_test_acc[1]
        
        best_test_loss = train_val_test_loss[2]
        best_test_acc = train_val_test_acc[2]
        
        best_embeddings = embeddings
        wait = 0
    else:
        if wait >= PATIENCE:
            print('Epoch {}: early stopping'.format(epoch))
            break
        wait += 1

#save metrics
loss_acc_log = open("{}/data/output/{}/embedding/{}/4L_dim_hidden_{}_p_val_{}/loss_acc_log.txt".format(project_path, FLAGS.dataset, FLAGS.embedding_method, FLAGS.embedding_size, FLAGS.p_value_threshold), "w")
metrics = "GCN embedding - 4 hidden layer - dim(embedding) = {} \n train_loss= {:.4f} \n train_acc= {:.4f} \n val_loss= {:.4f} \n val_acc= {:.4f} \n test_loss= {:.4f} \n test_acc= {:.4f}"\
            .format(FLAGS.embedding_size, best_train_loss, best_train_acc, best_val_loss, best_val_acc, best_test_loss, best_test_acc)
loss_acc_log.write(metrics)
loss_acc_log.close()
        
#Save the node emmbeddings
np.savetxt("{}/data/output/{}/embedding/{}/4L_dim_hidden_{}_p_val_{}/embeddings.txt".format(project_path, FLAGS.dataset, FLAGS.embedding_method, FLAGS.embedding_size, FLAGS.p_value_threshold), best_embeddings, delimiter="\t")
print("Embeddings saved in /data/output/{}/embedding/{}/4L_dim_hidden_{}_p_val_{}/embeddings.txt".format(FLAGS.dataset, FLAGS.embedding_method, FLAGS.embedding_size, FLAGS.p_value_threshold))
print("{}".format(metrics))
print("#################################################")