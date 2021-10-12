from __future__ import division
from __future__ import print_function

from pathlib import Path
import sys
project_path = Path(__file__).resolve().parents[1]

import scipy.sparse as sp
import os
import numpy as np
from scipy import stats
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
from sklearn.model_selection import train_test_split

# Set random seed
seed = 123

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_training_data(path="{}/data/parsed_input/".format(project_path), dataset="brc_microarray_usa"):
    """Load dataset"""
    print('Loading training {} dataset...'.format(dataset))

    data = np.genfromtxt("{}{}/training_data.txt".format(path, dataset), dtype=np.dtype(str))
    features = np.asarray(data[:-1, :-1], dtype=np.float32)
    labels = encode_onehot(data[-1, :-1])
    adj = sp.load_npz("{}/data/output/network/ppi.adjacency.npz".format(project_path))

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], adj.count_nonzero(), features.shape[0]))

    return features.T, adj, labels

def load_testing_data(path="{}/data/parsed_input/".format(project_path), dataset="brc_microarray_usa"):
    """Load dataset"""
    print('Loading testing {} dataset...'.format(dataset))

    data = np.genfromtxt("{}{}/testing_data.txt".format(path, dataset), dtype=np.dtype(str))
    features = np.asarray(data[:-1, :-1], dtype=np.float32)
    labels = encode_onehot(data[-1, :-1])
    adj = sp.load_npz("{}/data/output/network/ppi.adjacency.npz".format(project_path))

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], adj.count_nonzero(), features.shape[0]))

    return features.T, adj, labels

def load_extended_testing_data(path="{}/data/parsed_input/".format(project_path), dataset="brc_microarray_usa"):
    """Load dataset"""
    print('Loading extended testing {} dataset...'.format(dataset))

    data = np.genfromtxt("{}{}/extended_testing_data.txt".format(path, dataset), dtype=np.dtype(str))
    features = np.asarray(data[:-1, :-1], dtype=np.float32)
    labels = encode_onehot(data[-1, :-1])
    adj = sp.load_npz("{}/data/output/network/ppi.adjacency.npz".format(project_path))

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], adj.count_nonzero(), features.shape[0]))

    return features.T, adj, labels
    
def load_validation_data(path="{}/data/parsed_input/".format(project_path), dataset="brc_microarray_usa"):
    """Load dataset"""
    print('Loading validation {} dataset...'.format(dataset))

    data = np.genfromtxt("{}{}/validation_data.txt".format(path, dataset), dtype=np.dtype(str))
    features = np.asarray(data[:-1, :-1], dtype=np.float32)
    labels = encode_onehot(data[-1, :-1])
    adj = sp.load_npz("{}/data/output/network/ppi.adjacency.npz".format(project_path))

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], adj.count_nonzero(), features.shape[0]))

    return features.T, adj, labels

# For hyper-parameters tuning use all the available data
def load_training_data_tuning(path="{}/data/parsed_input/".format(project_path), dataset="brc_microarray_usa", p_val=0.1):
    """Load dataset"""
    print('Loading training {} dataset...'.format(dataset))

    data = np.genfromtxt("{}{}/p_val_{}/training_data.txt".format(path, dataset, p_val), dtype=np.dtype(str))
    features = np.asarray(data[:-1, :-1], dtype=np.float32)
    labels = encode_onehot(data[-1, :-1])
    adj = sp.load_npz("{}/data/output/network/ppi.adjacency.npz".format(project_path))

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], adj.count_nonzero(), features.shape[0]))

    return features.T, adj, labels

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def normalize_adj_numpy(adj, symmetric=True):
    if symmetric:
        d = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d)
    else:
        d = np.diag(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj)
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj


def preprocess_adj_numpy(adj, symmetric=True):
    adj = adj + np.eye(adj.shape[0])
    adj = normalize_adj_numpy(adj, symmetric)
    return adj


def preprocess_adj_tensor(adj_tensor, symmetric=True):
    adj_out_tensor = []
    for i in range(adj_tensor.shape[0]):
        adj = adj_tensor[i]
        adj = adj + np.eye(adj.shape[0])
        adj = normalize_adj_numpy(adj, symmetric)
        adj_out_tensor.append(adj)
    adj_out_tensor = np.array(adj_out_tensor)
    return adj_out_tensor


def preprocess_adj_tensor_with_identity(adj_tensor, symmetric=True):
    adj_out_tensor = []
    for i in range(adj_tensor.shape[0]):
        adj = adj_tensor[i]
        adj = adj + np.eye(adj.shape[0])
        adj = normalize_adj_numpy(adj, symmetric)
        adj = np.concatenate([np.eye(adj.shape[0]), adj], axis=0)
        adj_out_tensor.append(adj)
    adj_out_tensor = np.array(adj_out_tensor)
    return adj_out_tensor


def preprocess_adj_tensor_with_identity_concat(adj_tensor, symmetric=True):
    adj_out_tensor = []
    for i in range(adj_tensor.shape[0]):
        adj = adj_tensor[i]
        adj = adj + np.eye(adj.shape[0])
        adj = normalize_adj_numpy(adj, symmetric)
        adj = np.concatenate([np.eye(adj.shape[0]), adj], axis=0)
        adj_out_tensor.append(adj)
    adj_out_tensor = np.concatenate(adj_out_tensor, axis=0)
    return adj_out_tensor

def preprocess_adj_tensor_concat(adj_tensor, symmetric=True):
    adj_out_tensor = []
    for i in range(adj_tensor.shape[0]):
        adj = adj_tensor[i]
        adj = adj + np.eye(adj.shape[0])
        adj = normalize_adj_numpy(adj, symmetric)
        adj_out_tensor.append(adj)
    adj_out_tensor = np.concatenate(adj_out_tensor, axis=0)
    return adj_out_tensor

def preprocess_edge_adj_tensor(edge_adj_tensor, symmetric=True):
    edge_adj_out_tensor = []
    num_edge_features = int(edge_adj_tensor.shape[1]/edge_adj_tensor.shape[2])

    for i in range(edge_adj_tensor.shape[0]):
        edge_adj = edge_adj_tensor[i]
        edge_adj = np.split(edge_adj, num_edge_features, axis=0)
        edge_adj = np.array(edge_adj)
        edge_adj = preprocess_adj_tensor_concat(edge_adj, symmetric)
        edge_adj_out_tensor.append(edge_adj)

    edge_adj_out_tensor = np.array(edge_adj_out_tensor)
    return edge_adj_out_tensor


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

#===============================================================================
#Split the data into 0.6 for training, 0.2 for validation, and 0.2 for testing
# (Mask the labels of missing gene expressions)
#===============================================================================
def get_splits_for_training(y, dataset="brc_microarray_usa"):
    labels = np.argmax(y, axis=1)
    indices = np.arange(y.shape[0])
    #Initialize the splits
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    #From the first split get only the test set
    y_train_tmp, _, labels_train_tmp, _, idx_train_tmp, idx_test = train_test_split(y, labels, indices, test_size=0.2, stratify=labels, random_state=seed) #test: 0.2, tmp_train: 0.8
    y_test[idx_test] = y[idx_test]
    #Split the temporary train set to get the final train and val sets
    _, _, _, _, idx_train, idx_val = train_test_split(y_train_tmp, labels_train_tmp, idx_train_tmp, test_size=0.25, stratify=labels_train_tmp, random_state=seed) # val: 0.2, train: 0.6 (0.25 x 0.8 = 0.2)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    #Remove the indices of missing gene expressions form the mask
    if os.path.isfile("{}/data/parsed_input/{}/missing_ge_idx.txt".format(project_path, dataset)):
        missing_ge_idx = np.genfromtxt("{}/data/parsed_input/{}/missing_ge_idx.txt".format(project_path, dataset), dtype=np.int32)
    else:
        missing_ge_idx = []
    idx_train_mask = [item for item in idx_train if item not in missing_ge_idx]
    #Get the mask for the training set
    train_mask = sample_mask(idx_train_mask, y.shape[0])
    
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask

#===============================================================================
# Split the data into 0.2 for validation, 0.2 for testing
# but use all the data for training to generate the final embeddings
# (Mask the labels of missing gene expressions)
#===============================================================================
def get_splits_for_learning(y, dataset="brc_microarray_usa"):
    labels = np.argmax(y, axis=1)
    indices = np.arange(y.shape[0])
    #Initialize the splits
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    #From the first split get only the test set
    y_train_tmp, _, labels_train_tmp, _, idx_train_tmp, idx_test = train_test_split(y, labels, indices, test_size=0.2, stratify=labels, random_state=seed) #test: 0.2, tmp_train: 0.8
    y_test[idx_test] = y[idx_test]
    #Split the temporary train set to get the final train and val sets
    _, _, _, _, idx_train, idx_val = train_test_split(y_train_tmp, labels_train_tmp, idx_train_tmp, test_size=0.25, stratify=labels_train_tmp, random_state=seed) # val: 0.2, train: 0.6 (0.25 x 0.8 = 0.2)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    
    #Override the training split and replace it with all available data
    y_train = y
    idx_train = indices
    #Remove the indices of missing gene expressions form the mask
    if os.path.isfile("{}/data/parsed_input/{}/missing_ge_idx.txt".format(project_path, dataset)):
        missing_ge_idx = np.genfromtxt("{}/data/parsed_input/{}/missing_ge_idx.txt".format(project_path, dataset), dtype=np.int32)
    else:
        missing_ge_idx = []
    idx_train_mask = [item for item in idx_train if item not in missing_ge_idx]
    #Get the mask for the training set
    train_mask = sample_mask(idx_train_mask, y.shape[0])
    
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask

#===============================================================================
#Split the data into 0.6 for training, 0.2 for validation, and 0.2 for testing
# (Mask the labels of missing gene expressions)
#===============================================================================
def get_splits_for_training_tuning(y, dataset="brc_microarray_usa", p_val=0.1):
    labels = np.argmax(y, axis=1)
    indices = np.arange(y.shape[0])
    #Initialize the splits
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    #From the first split get only the test set
    y_train_tmp, _, labels_train_tmp, _, idx_train_tmp, idx_test = train_test_split(y, labels, indices, test_size=0.2, stratify=labels, random_state=seed) #test: 0.2, tmp_train: 0.8
    y_test[idx_test] = y[idx_test]
    #Split the temporary train set to get the final train and val sets
    _, _, _, _, idx_train, idx_val = train_test_split(y_train_tmp, labels_train_tmp, idx_train_tmp, test_size=0.25, stratify=labels_train_tmp, random_state=seed) # val: 0.2, train: 0.6 (0.25 x 0.8 = 0.2)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    #Remove the indices of missing gene expressions form the mask
    if os.path.isfile("{}/data/parsed_input/{}/p_val_{}/missing_ge_idx.txt".format(project_path, dataset, p_val)):
        missing_ge_idx = np.genfromtxt("{}/data/parsed_input/{}/p_val_{}/missing_ge_idx.txt".format(project_path, dataset, p_val), dtype=np.int32)
    else:
        missing_ge_idx = []
    idx_train_mask = [item for item in idx_train if item not in missing_ge_idx]
    #Get the mask for the training set
    train_mask = sample_mask(idx_train_mask, y.shape[0])
    
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask

#===============================================================================
# Split the data into 0.2 for validation, 0.2 for testing
# but use all the data for training to generate the final embeddings
# (Mask the labels of missing gene expressions)
#===============================================================================
def get_splits_for_learning_tuning(y, dataset="brc_microarray_usa", p_val=0.1):
    labels = np.argmax(y, axis=1)
    indices = np.arange(y.shape[0])
    #Initialize the splits
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    #From the first split get only the test set
    y_train_tmp, _, labels_train_tmp, _, idx_train_tmp, idx_test = train_test_split(y, labels, indices, test_size=0.2, stratify=labels, random_state=seed) #test: 0.2, tmp_train: 0.8
    y_test[idx_test] = y[idx_test]
    #Split the temporary train set to get the final train and val sets
    _, _, _, _, idx_train, idx_val = train_test_split(y_train_tmp, labels_train_tmp, idx_train_tmp, test_size=0.25, stratify=labels_train_tmp, random_state=seed) # val: 0.2, train: 0.6 (0.25 x 0.8 = 0.2)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    
    #Override the training split and replace it with all available data
    y_train = y
    idx_train = indices
    #Remove the indices of missing gene expressions form the mask
    if os.path.isfile("{}/data/parsed_input/{}/p_val_{}/missing_ge_idx.txt".format(project_path, dataset, p_val)):
        missing_ge_idx = np.genfromtxt("{}/data/parsed_input/{}/p_val_{}/missing_ge_idx.txt".format(project_path, dataset, p_val), dtype=np.int32)
    else:
        missing_ge_idx = []
    idx_train_mask = [item for item in idx_train if item not in missing_ge_idx]
    #Get the mask for the training set
    train_mask = sample_mask(idx_train_mask, y.shape[0])
    
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask

def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))


def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))


def evaluate_preds(preds, labels, indices):
    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc


def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian


def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k + 1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
