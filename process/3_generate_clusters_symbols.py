from __future__ import division
from __future__ import print_function

from pathlib import Path
import sys
project_path = Path(__file__).resolve().parents[1]
sys.path.append(str(project_path))

import tensorflow as tf
import os
import scipy.sparse as sp
import numpy as np
from core.GeometricAffinityPropagation import AffinityPropagation
from sklearn.metrics import euclidean_distances
import mygene
from collections import defaultdict
import csv

# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'brc_microarray_usa', 'Dataset string.')

flags.DEFINE_string('clustering_method', 'geometric_ap', 'Name of output folder for the clustering method.')
flags.DEFINE_string('embedding_method', 'gcn', 'Name of the embedding method.')
flags.DEFINE_string('clusters_output', 'clusters.txt', 'Name of desired output file of obtained clusters.')
flags.DEFINE_string('clusters_symbols', 'clusters.symbols.txt', 'Name of output file of ensembl names of cluster genes.')

#Check data availability    
if not os.path.isfile("{}/data/output/{}/clustering/{}/{}/{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.embedding_method, FLAGS.clusters_output)):
    sys.exit("{} file is not available under /data/output/{}/clustering/{}/{}/".format(FLAGS.clusters_output, FLAGS.dataset, FLAGS.clustering_method, FLAGS.embedding_method))

print("----------------------------------------")
print("----------------------------------------")
print("Clustering configuration:")
print("Dataset: {}".format(FLAGS.dataset))
print("Clustering Method: {}".format(FLAGS.clustering_method))
print("Embedding Method: {}".format(FLAGS.embedding_method))
print("----------------------------------------")
print("----------------------------------------")
    
def prepare_clusters_ensembl_symbols(clusters_file="clusters.txt"):
    
    Ids = np.genfromtxt("{}/data/output/network/ppi.ids.txt".format(project_path), dtype=np.dtype(str), delimiter="\t")
    protein_ids = {}
    ids_protein = {}
    for i in range(Ids.shape[0]):
        protein_ids[Ids[i,0]]=Ids[i,1]
        ids_protein[Ids[i,1]]=Ids[i,0]
            
    mg = mygene.MyGeneInfo()
    map_ensp_symbol={}
    print("Request gene symbols by gene query web service...")
    annotations = mg.querymany(protein_ids.keys(), scopes='ensembl.protein', fields='symbol', species='human')
    #For each query map ENSPs to the gene symbol
    for response in annotations:
        ensp = response['query']
        if('symbol' in response):
            map_ensp_symbol[ensp] = response['symbol']
        else:
            map_ensp_symbol[ensp] = ensp
                    
    Clusters = open("{}/data/output/{}/clustering/{}/{}/{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.embedding_method, clusters_file), encoding="utf-8")
    with open("{}/data/output/{}/clustering/{}/{}/{}".format(project_path, FLAGS.dataset, FLAGS.clustering_method, FLAGS.embedding_method, FLAGS.clusters_symbols), "w", newline='', encoding="utf-8") as f:
        w_cluster = csv.writer(f, delimiter ='\t')
        for line in Clusters:
            line = line.strip()
            columns = line.split("\t")
            cl = []
            for i in range(1, len(columns)): #Skip first column that contains the exemplar
                if(ids_protein[columns[i]] in map_ensp_symbol.keys()):
                    cl.append(map_ensp_symbol[ids_protein[columns[i]]])
            w_cluster.writerow(cl)
    Clusters.close()
    
print("Preparing gene symbols for saved clusters ...")
prepare_clusters_ensembl_symbols(clusters_file=FLAGS.clusters_output)
print("Clusters with gene symbols saved")
