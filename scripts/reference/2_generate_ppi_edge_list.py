from __future__ import division
from __future__ import print_function

import tensorflow as tf
from pathlib import Path
import numpy as np
from collections import defaultdict
import csv
import os
import sys
project_path = Path(__file__).resolve().parents[2]


# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('network', '9606.protein.links.v11.0.txt', 'Name of network file.')  # 'PPI'
flags.DEFINE_string('node_ids', 'ppi.ids.txt', 'Ensemble unique ids file.')

#Check data availability
if not os.path.isfile("{}/data/reference/ppi_network/{}".format(project_path, FLAGS.network)):
    sys.exit("{} network is not available under /data/reference/ppi_network/".format(FLAGS.network))
    
if not os.path.isfile("{}/data/output/network/{}".format(project_path, FLAGS.node_ids)):
    sys.exit("{} mapping file is not available under /data/output/network/".format(FLAGS.node_ids))
    
print("Generate edge list network...")
#Read network
PPI = np.genfromtxt("{}/data/reference/ppi_network/{}".format(project_path, FLAGS.network), skip_header=1, usecols=(0,1), dtype=np.dtype(str))

protein_ids = {}
with open("{}/data/output/network/{}".format(project_path, FLAGS.node_ids)) as f:
    for line in f:
       (key, val) = line.split()
       protein_ids[key] = val
       
edgelist = defaultdict(list)
for i in range(PPI.shape[0]):
    ENSP1 = PPI[i][0].split(".",1)[1]
    ENSP2 = PPI[i][1].split(".",1)[1]
    edgelist[protein_ids[ENSP1]].append(protein_ids[ENSP2])        

with open("{}/data/output/network/ppi.edgelist.txt".format(project_path), "w", newline='', encoding="utf-8") as f:
    w_adj = csv.writer(f, delimiter ='\t')
    for key, val in edgelist.items():
        for neighbor in val:
            w_adj.writerow([key, neighbor])
                
print("Successful generation of edge list network in /data/output/network/ppi.edgelist.txt")