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
flags.DEFINE_string('dataset', 'brc_microarray_netherlands', 'Dataset string.')
flags.DEFINE_string('gene_expression', 'Table_NKI_295', 'Gene expression prefix.')
flags.DEFINE_string('clinical_info', 'Table1_ClinicalData_Table.xls', 'Clinical status file.')
flags.DEFINE_string('node_ids', 'ppi.ids.txt', 'Ensembl unique ids file.')
flags.DEFINE_list('labels', ["free", "metastasis"], 'List of class labels.')

clinical_columns = (1,4)
clinical_skip_header= 3
ge_skip_header = 2

#Check data availability
if not os.path.isdir("{}/data/raw_input/{}".format(project_path, FLAGS.dataset)):
    os.makedirs("{}/data/raw_input/{}".format(project_path, FLAGS.dataset))

if not os.path.isdir("{}/data/parsed_input/{}".format(project_path, FLAGS.dataset)):
    os.makedirs("{}/data/parsed_input/{}".format(project_path, FLAGS.dataset))
    
if not os.path.isfile("{}/data/output/network/{}".format(project_path, FLAGS.node_ids)):
    sys.exit("{} mapping file is not available under /data/output/network/".format(FLAGS.node_ids))

print("Generate gene expression matrix...")    

sample_clinical = {}
clinical_info = pd.read_excel("{}/data/raw_input/{}/{}".format(project_path, FLAGS.dataset, FLAGS.clinical_info), usecols=clinical_columns, skiprows=clinical_skip_header)
for i,j in zip(clinical_info["SampleID"], clinical_info["EVENTmeta"]):
    sample_clinical[i] = j
    
     
clinical_status = clinical_info["EVENTmeta"].to_numpy(dtype=np.dtype(str))
clinical_status[clinical_status == "0"] = FLAGS.labels[0]
clinical_status[clinical_status == "1"] = FLAGS.labels[1]

size_samples = [50, 50, 50, 50, 50, 45]
ge_offset = 2
sz_column = 5
GE_files = OrderedDict()
start_clinical_status = 0
for num_file in range(len(size_samples)):
    #Each sample has 5 columns
    #We select columns 1 (log ratio) and 5 (flag) of each sample
    #Note that first two columns are for gene names
    selected_flags = [i for i in range(ge_offset,ge_offset+(size_samples[num_file]*sz_column)) if i%5 == 1]        
    selected_columns = [i for i in range(ge_offset,ge_offset+(size_samples[num_file]*sz_column)) if i%5 == 2]
    selected_ids = [i for i in range(ge_offset,ge_offset+(size_samples[num_file]*sz_column)) if i%5 == 2]
    idx_probe = 0
    selected_columns.insert(0,idx_probe) #Add index of first column that contains the probe
    GE_unfiltered = np.genfromtxt("{}/data/raw_input/{}/{}_{}.txt".format(project_path, FLAGS.dataset, FLAGS.gene_expression,num_file+1), skip_header=ge_skip_header, usecols=tuple(selected_columns), dtype=np.dtype(str), delimiter="\t")
    spot_flags = np.genfromtxt("{}/data/raw_input/{}/{}_{}.txt".format(project_path, FLAGS.dataset, FLAGS.gene_expression, num_file+1), skip_header=ge_skip_header, usecols=tuple(selected_flags), dtype=np.int32, delimiter="\t")
    
    #Make sure the samples are correctly labeled
    file_samples = np.genfromtxt("{}/data/raw_input/{}/{}_{}.txt".format(project_path, FLAGS.dataset, FLAGS.gene_expression, num_file+1), max_rows=1, usecols=tuple(selected_ids), dtype=np.dtype(str), delimiter="\t")
    file_samples = [id.replace('Sample ', '') for id in file_samples]
    sample_labels = [FLAGS.labels[sample_clinical[int(id)]] for id in file_samples]
    ref_array = clinical_status[start_clinical_status:start_clinical_status+size_samples[num_file]]
    np.testing.assert_array_equal(sample_labels, ref_array)
    start_clinical_status += size_samples[num_file]
    
    valid_measurement_idx = np.where(np.all(spot_flags == 1, axis=1))
    expression_matrix = np.squeeze(GE_unfiltered[valid_measurement_idx,:])
    GE_files[num_file] = pd.DataFrame(data=expression_matrix[:,1:], index=expression_matrix[:,0])

GE_df = pd.concat([GE_files[0], GE_files[1], GE_files[2], GE_files[3], GE_files[4], GE_files[5]], axis=1, join='inner')
                                  
probes = GE_df.index.to_numpy()

mg = mygene.MyGeneInfo()
  
map_ensembl={}
# Check the available fields from:
#https://docs.mygene.info/en/latest/doc/data.html#available-fields
annotations = mg.querymany(probes, scopes='accession, ensembl', fields='ensembl.protein', species='human')
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

ge_dict = GE_df.T.to_dict('list')#Make the gene expression a dict with the probes as keys

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