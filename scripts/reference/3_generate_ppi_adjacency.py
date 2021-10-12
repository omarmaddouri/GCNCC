from __future__ import division
from __future__ import print_function

import tensorflow as tf
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import os
import sys
project_path = Path(__file__).resolve().parents[2]

# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('edgelist', 'ppi.edgelist.txt', 'Edgelist file.')  # 'PPI'

#Check data availability
if not os.path.isfile("{}/data/output/network/{}".format(project_path, FLAGS.edgelist)):
    sys.exit("{} file is not available under /data/output/network/".format(FLAGS.edgelist))
    
print("Generate adjacency matrix...")
# build graph
ppi_ids = np.genfromtxt("{}/data/output/network/ppi.ids.txt".format(project_path), dtype=np.dtype(str))
idx = np.array(ppi_ids[:, 1], dtype=np.int32)

idx_map = {j: i for i, j in enumerate(idx)}
edges_unordered = np.genfromtxt("{}/data/output/network/ppi.edgelist.txt".format(project_path), dtype=np.int32)
edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                 dtype=np.int32).reshape(edges_unordered.shape)
adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(idx.shape[0], idx.shape[0]), dtype=np.float32)

# build symmetric adjacency matrix
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
sp.save_npz("{}/data/output/network/ppi.adjacency.npz".format(project_path), adj)
print("Successful generation of adjacency matrix in /data/output/network/ppi.adjacency.npz")