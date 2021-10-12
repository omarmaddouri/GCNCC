from __future__ import division
from __future__ import print_function

import tensorflow as tf
from pathlib import Path
import numpy as np
import csv
import os
import sys
project_path = Path(__file__).resolve().parents[2]


# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('network', '9606.protein.links.v11.0.txt', 'Name of network file.')  # 'PPI'

#Check dataset availability
if not os.path.isfile("{}/data/reference/ppi_network/{}".format(project_path, FLAGS.network)):
    sys.exit("{} network is not available under /data/reference/ppi_network/".format(FLAGS.network))

if not os.path.isdir("{}/data/output/network".format(project_path)):
    os.makedirs("{}/data/output/network".format(project_path))
        
print("Generate unique ids to the network node entities...")
#Read network
PPI = np.genfromtxt("{}/data/reference/ppi_network/{}".format(project_path, FLAGS.network), skip_header=1, usecols=(0,1), dtype=np.dtype(str))
protein_ids = {}
p_id = 0
#Assign int IDs to ensemble IDs
for i in range(PPI.shape[0]):
    ENSP1 = PPI[i][0].split(".",1)[1]
    ENSP2 = PPI[i][1].split(".",1)[1]
    if(ENSP1 not in protein_ids):
        protein_ids[ENSP1] = p_id
        p_id+=1
    if(ENSP2 not in protein_ids):
        protein_ids[ENSP2] = p_id
        p_id+=1
with open("{}/data/output/network/ppi.ids.txt".format(project_path), "w", newline='', encoding="utf-8") as f:
    w_indices = csv.writer(f, delimiter ='\t')
    for key, val in protein_ids.items():
        w_indices.writerow([key, val])

print("Successful generation of network node ids in /data/output/network/ppi.ids.txt")