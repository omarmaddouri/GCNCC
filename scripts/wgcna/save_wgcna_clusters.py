from __future__ import division
from __future__ import print_function

from pathlib import Path
import sys
project_path = Path(__file__).resolve().parents[2]
sys.path.append(str(project_path))

import tensorflow as tf
import os
import numpy as np
import csv
from collections import defaultdict

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'brc_microarray_usa', 'Dataset string.')
flags.DEFINE_string('clustering_method', 'wgcna', 'Name of output folder for the clustering method.')
flags.DEFINE_string('embedding_method', 'ge', 'Name of the embedding method.')

dict = defaultdict(list)

WGCNA_clust = np.genfromtxt("{}/data/output/{}/clustering/{}/{}/clusters_wgcna/final-membership.txt".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.embedding_method), skip_header=1, dtype=np.dtype(str))

for i in range(WGCNA_clust.shape[0]):
    if (WGCNA_clust[i,1] != "UNCLASSIFIED"):
        dict[WGCNA_clust[i,1]].append(int(WGCNA_clust[i,0]))
    
with open("{}/data/output/{}/clustering/{}/{}/clusters.txt".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.embedding_method), "w", newline='', encoding="utf-8") as f:
    w_clusters = csv.writer(f, delimiter ='\t')
    for key, val in dict.items():
        line = []
        line.append(key)
        for item in val:
            line.append(item)
        w_clusters.writerow(line)

print("WGCNA clusters saved.")