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
from sklearn.preprocessing import normalize

# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'scz_microarray_combined', 'Dataset string.')
flags.DEFINE_string('gene_expression', '_series_matrix.txt', 'Gene expression suffix.')
flags.DEFINE_string('node_ids', 'ppi.ids.txt', 'Ensembl unique ids file.')
flags.DEFINE_list('labels', ["control", "schizophrenia"], 'List of class labels.')

GSE_files = ["GSE12649", "GSE21138", "GSE53987"]

#Check data availability
if not os.path.isdir("{}/data/raw_input/{}".format(project_path, FLAGS.dataset)):
    os.makedirs("{}/data/raw_input/{}".format(project_path, FLAGS.dataset))
    
if not os.path.isdir("{}/data/parsed_input/{}".format(project_path, FLAGS.dataset)):
    os.makedirs("{}/data/parsed_input/{}".format(project_path, FLAGS.dataset))

for gse in GSE_files:
    if not os.path.isfile("{}/data/raw_input/{}/{}{}".format(project_path, FLAGS.dataset, gse, FLAGS.gene_expression)):
        sys.exit("{}{} file is not available under /data/raw_input/{}".format(gse, FLAGS.gene_expression, FLAGS.dataset))
    
if not os.path.isfile("{}/data/output/network/{}".format(project_path, FLAGS.node_ids)):
    sys.exit("{} mapping file is not available under /data/output/network/".format(FLAGS.node_ids))

print("Generate gene expression matrix...")    

labels_row = [34, 36, 28]
control_string = ["control", "Control", "control"]
scz_string = ["schizophrenia", "Scz", "schiz"]
skip_header = [70, 75, 67]
skip_footer = [1, 1, 1]

combined_clinical_status = []
GE_dict = OrderedDict()
for i in range( len(GSE_files) ):
    with open("{}/data/raw_input/{}/{}{}".format(project_path, FLAGS.dataset, GSE_files[i], FLAGS.gene_expression), encoding="utf-8") as lines:
        clinical_status = np.genfromtxt(itertools.islice(lines, labels_row[i]-1, labels_row[i]), delimiter="\t", dtype=np.dtype(str))
    for j in range(len(clinical_status)):
        if(clinical_status[j].find(control_string[i]) != -1):
            clinical_status[j] = FLAGS.labels[0]
        elif(clinical_status[j].find(scz_string[i]) != -1):
            clinical_status[j] = FLAGS.labels[1]
            
    indices = np.where(np.logical_or(clinical_status == FLAGS.labels[0], clinical_status == FLAGS.labels[1]))
    selected_columns = [y for x in indices for y in x]
    clinical_status = clinical_status[selected_columns] #Keep only labels of interest
    idx_probe = 0
    selected_columns.insert(0,idx_probe) #Add index of first column that contains the probe
    expression_matrix = np.genfromtxt("{}/data/raw_input/{}/{}{}".format(project_path, FLAGS.dataset, GSE_files[i], FLAGS.gene_expression), skip_header=skip_header[i], skip_footer=skip_footer[i], usecols=tuple(selected_columns), dtype=np.dtype(str), delimiter="\t")
    #Normalize the data in each dataset
    expression_matrix[:,1:] = normalize(expression_matrix[:,1:], norm='l1')
    GE_dict[i] = pd.DataFrame(data=expression_matrix[:,1:], index=expression_matrix[:,0])
    combined_clinical_status += list(clinical_status)

combined_GE = pd.concat([GE_dict[0], GE_dict[1], GE_dict[2]], axis=1, join='inner')
combined_GE.index = combined_GE.index.map(lambda x: x.replace('"', ''))

probes = combined_GE.index.to_numpy()

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

ge_dict = combined_GE.T.to_dict('list')#Make the gene expression a dict with the probes as keys

shape_ge = (len(protein_ids), len(combined_clinical_status))
ge_matrix = np.zeros(shape_ge)

for i in range(ge_matrix.shape[0]):
    if(protein_ids[i] in map_ensembl and map_ensembl[protein_ids[i]][0] in ge_dict):
        ge_matrix[i,:] = ge_dict[map_ensembl[protein_ids[i]][0]]

feature_label = np.zeros((ge_matrix.shape[1], ge_matrix.shape[0]+1), dtype=object) #Additional column for labels
feature_label[:,:-1] = ge_matrix.T
feature_label[:,-1] = combined_clinical_status

np.savetxt("{}/data/parsed_input/{}/feature_label.txt".format(project_path, FLAGS.dataset), feature_label, fmt="%s")
print("Successful generation of feature_label matrix")