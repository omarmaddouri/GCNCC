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
flags.DEFINE_string('dataset', 'scz_rnaseq_syn2759792', 'Dataset string.')
flags.DEFINE_string('gene_expression', 'CMC_MSSM-Penn-Pitt_DLPFC_mRNA_IlluminaHiSeq2500_geneExpressionRaw.tsv', 'Gene expression file.')
flags.DEFINE_string('clinical_info', 'CMC_MSSM-Penn-Pitt_Clinical.csv', 'Clinical status file.')
flags.DEFINE_string('node_ids', 'ppi.ids.txt', 'Ensembl unique ids file.')
flags.DEFINE_list('labels', ["control", "schizophrenia"], 'List of class labels.')

#Check data availability
if not os.path.isdir("{}/data/raw_input/{}".format(project_path, FLAGS.dataset)):
    os.makedirs("{}/data/raw_input/{}".format(project_path, FLAGS.dataset))

if not os.path.isdir("{}/data/parsed_input/{}".format(project_path, FLAGS.dataset)):
    os.makedirs("{}/data/parsed_input/{}".format(project_path, FLAGS.dataset))
        
if not os.path.isfile("{}/data/raw_input/{}/{}".format(project_path, FLAGS.dataset, FLAGS.gene_expression)):
    sys.exit("{} file is not available under /data/raw_input/{}".format(FLAGS.gene_expression, FLAGS.dataset))
    
if not os.path.isfile("{}/data/output/network/{}".format(project_path, FLAGS.node_ids)):
    sys.exit("{} mapping file is not available under /data/output/network/".format(FLAGS.node_ids))

print("Generate gene expression matrix...")    
clinical_info = pd.read_csv("{}/data/raw_input/{}/{}".format(project_path, FLAGS.dataset, FLAGS.clinical_info), usecols=(2,16)).to_numpy(dtype=np.dtype(str))
sample_labels = {}
for i in range(clinical_info.shape[0]):
    sample_labels[clinical_info[i,1]] = clinical_info[i,0]

with open("{}/data/raw_input/{}/{}".format(project_path, FLAGS.dataset, FLAGS.gene_expression), encoding="utf-8") as lines:
    clinical_status = np.genfromtxt(itertools.islice(lines, 0, 1), delimiter="\t", dtype=np.dtype(str))
for i in range(len(clinical_status)):
    if(clinical_status[i] in sample_labels.keys()):
        clinical_status[i] = sample_labels[clinical_status[i]]
    if(clinical_status[i].find("Control") != -1):
        clinical_status[i] = FLAGS.labels[0]
    elif(clinical_status[i].find("SCZ") != -1):
        clinical_status[i] = FLAGS.labels[1]
        
indices = np.where(np.logical_or(clinical_status == FLAGS.labels[0], clinical_status == FLAGS.labels[1]))
selected_columns = [y for x in indices for y in x]
clinical_status = clinical_status[selected_columns] #Keep only labels of interest
idx_probe = 0
#Increment idx of selected columns to account for the gene ids column
selected_columns = [x+1 for x in selected_columns]

selected_columns.insert(0,idx_probe) #Add index of first column that contains the probe
GE = np.genfromtxt("{}/data/raw_input/{}/{}".format(project_path, FLAGS.dataset, FLAGS.gene_expression), skip_header=1, usecols=tuple(selected_columns), dtype=np.dtype(str), delimiter="\t")
probes = GE[:,0]

mg = mygene.MyGeneInfo()
  
map_ensembl={}
annotations = mg.querymany(probes, scopes='ensembl.gene', fields='ensembl.protein', species='human')
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
    ge_dict[GE[j,0].replace('"', '')] = GE[j,1:] #Make the gene expression a dict with the probes as keys

shape_ge = (len(protein_ids), len(clinical_status))
ge_matrix = np.zeros(shape_ge)

for i in range(ge_matrix.shape[0]):
    if(protein_ids[i] in map_ensembl and map_ensembl[protein_ids[i]][0] in ge_dict):
        ge_matrix[i,:] = ge_dict[map_ensembl[protein_ids[i]][0]]

feature_label = np.zeros((ge_matrix.shape[1], ge_matrix.shape[0]+1), dtype=object) #Additional column for labels
feature_label[:,:-1] = ge_matrix.T
feature_label[:,-1] = clinical_status

np.savetxt("{}/data/parsed_input/{}/feature_label.txt".format(project_path, FLAGS.dataset), feature_label, fmt="%s")
print("Successful generation of feature_label matrix")