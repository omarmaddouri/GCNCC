from __future__ import division
from __future__ import print_function

import tensorflow as tf
from pathlib import Path
import numpy as np
import pandas as pd
from collections import OrderedDict
import itertools
import csv
import os
import sys
project_path = Path(__file__).resolve().parents[2]
sys.path.append(str(project_path))
import mygene

# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'scz_microarray_combined', 'Dataset string.')
flags.DEFINE_string('gene_expression', 'datExpr', 'Gene expression file.')
flags.DEFINE_string('clinical_info', 'datMeta', 'Clinical status file.')
flags.DEFINE_string('node_ids', 'ppi.ids.txt', 'Ensembl unique ids file.')
flags.DEFINE_list('labels', ["control", "schizophrenia"], 'List of class labels.')

GSE_files = ["_GSE12649.csv", "_GSE21138.csv", "_GSE53987.csv"]
GSE_labels = [["Control", "Schizophrenia"], ["CTL", "SCZ"], ["CTL", "SCZ"]]
GSE_cols = [(0,9), (0,3), (0,15)]

#Check data availability
if not os.path.isdir("{}/data/raw_input/{}".format(project_path, FLAGS.dataset)):
    os.makedirs("{}/data/raw_input/{}".format(project_path, FLAGS.dataset))

if not os.path.isdir("{}/data/parsed_input/{}".format(project_path, FLAGS.dataset)):
    os.makedirs("{}/data/parsed_input/{}".format(project_path, FLAGS.dataset))

for gse_index in range(len(GSE_files)):
    gse_file = GSE_files[gse_index]        
    if not os.path.isfile("{}/data/raw_input/{}/{}{}".format(project_path, FLAGS.dataset, FLAGS.gene_expression,gse_file)):
        sys.exit("{}{} file is not available under /data/raw_input/{}".format(FLAGS.gene_expression, gse_file, FLAGS.dataset))
    
if not os.path.isfile("{}/data/output/network/{}".format(project_path, FLAGS.node_ids)):
    sys.exit("{} mapping file is not available under /data/output/network/".format(FLAGS.node_ids))
combined_feature_label_matrices = []
for gse_index in range(len(GSE_files)):
    gse_file = GSE_files[gse_index]
    print("Generate gene expression matrix...")    
    clinical_info = pd.read_csv("{}/data/raw_input/{}/{}{}".format(project_path, FLAGS.dataset, FLAGS.clinical_info, gse_file), usecols=GSE_cols[gse_index]).to_numpy(dtype=np.dtype(str))
    column_names = []
    clinical_status = []
    for i in range(clinical_info.shape[0]):
        if(clinical_info[i,1] in GSE_labels[gse_index]):
            column_names.append(clinical_info[i,0])
            clinical_status.append(clinical_info[i,1])
    GE = pd.read_csv("{}/data/raw_input/{}/{}{}".format(project_path, FLAGS.dataset, FLAGS.gene_expression, gse_file), names=column_names, header=0).to_numpy(dtype=np.dtype(str))
    probes = pd.read_csv("{}/data/raw_input/{}/{}{}".format(project_path, FLAGS.dataset, FLAGS.gene_expression, gse_file), usecols=[0], header=0).to_numpy(dtype=np.dtype(str))
    probes = probes.ravel()
    mg = mygene.MyGeneInfo()
      
    map_ensembl={}
    annotations = mg.querymany(probes, scopes='reporter', fields='ensembl.protein', species='human')
    #For each query map ENSPs to the reporter with highest score
    for response in annotations:
        if('ensembl' in response):
            matching_score = response['_score']
            scope = response['query']
            if(isinstance(response['ensembl'],list)):
                for prot_dict in response['ensembl']:
                    if(isinstance(prot_dict['protein'],list)):
                        for ensp in prot_dict['protein']:
                            if ensp in map_ensembl:
                                if(scope not in map_ensembl[ensp]):
                                    map_ensembl[ensp] = [scope, matching_score]
                                else:
                                    if(map_ensembl[ensp][1] < matching_score):
                                        map_ensembl[ensp] = [scope, matching_score]
                            else:
                                map_ensembl[ensp] = [scope, matching_score]
                    else:
                        ensp = prot_dict['protein']
                        if ensp in map_ensembl:
                            if(scope not in map_ensembl[ensp]):
                                map_ensembl[ensp] = [scope, matching_score]
                            else:
                                if(map_ensembl[ensp][1] < matching_score):
                                    map_ensembl[ensp] = [scope, matching_score]
                        else:
                            map_ensembl[ensp] = [scope, matching_score]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
            elif(isinstance(response['ensembl'],dict)):
                prot_dict = response['ensembl']
                if(isinstance(prot_dict['protein'],list)):
                    for ensp in prot_dict['protein']:
                        if ensp in map_ensembl:
                            if(scope not in map_ensembl[ensp]):
                                map_ensembl[ensp] = [scope, matching_score]
                            else:
                                if(map_ensembl[ensp][1] < matching_score):
                                    map_ensembl[ensp] = [scope, matching_score]
                        else:
                            map_ensembl[ensp] = [scope, matching_score]
                else:
                    ensp = prot_dict['protein']
                    if ensp in map_ensembl:
                        if(scope not in map_ensembl[ensp]):
                            map_ensembl[ensp] = [scope, matching_score]
                        else:
                            if(map_ensembl[ensp][1] < matching_score):
                                map_ensembl[ensp] = [scope, matching_score]
                    else:
                        map_ensembl[ensp] = [scope, matching_score]
    protein_ids = {}
    with open("{}/data/output/network/{}".format(project_path, FLAGS.node_ids)) as f:
        for line in f:
           (val, key) = line.split() #Read the ID as key
           protein_ids[int(key)] = val
    
    ge_dict = {}
    for j in range(GE.shape[0]):
        ge_dict[probes[j]] = GE[j,:] #Make the gene expression a dict with the probes as keys
    
    shape_ge = (len(protein_ids), len(clinical_status))
    ge_matrix = np.zeros(shape_ge)
    
    for i in range(ge_matrix.shape[0]):
        if(protein_ids[i] in map_ensembl and map_ensembl[protein_ids[i]][0] in ge_dict):
            ge_matrix[i,:] = ge_dict[map_ensembl[protein_ids[i]][0]]
    
    feature_label = np.zeros((ge_matrix.shape[1], ge_matrix.shape[0]+1), dtype=object) #Additional column for labels
    feature_label[:,:-1] = ge_matrix.T
    for i in range(len(clinical_status)):
        if(clinical_status[i].find(GSE_labels[gse_index][0]) != -1):
            clinical_status[i] = FLAGS.labels[0]
        elif(clinical_status[i].find(GSE_labels[gse_index][1]) != -1):
            clinical_status[i] = FLAGS.labels[1]
    feature_label[:,-1] = clinical_status
    combined_feature_label_matrices.append(feature_label)
feature_label = np.vstack((combined_feature_label_matrices[0], combined_feature_label_matrices[1], combined_feature_label_matrices[2]))
np.savetxt("{}/data/parsed_input/{}/feature_label.txt".format(project_path, FLAGS.dataset), feature_label, fmt="%s")
print("Successful generation of feature_label matrix")