#!/bin/bash
module load Python/3.6.6-intel-2018b
cd $SCRATCH/gcncc.bioinformatics                    # Go to project directory
source venv/bin/activate                            # Activate virtual environment (Command line should show environment name on left)
## Prepare data files
python -u scripts/brc_microarray_usa/1_prepare_feature_label_matrix.py
python -u scripts/brc_microarray_netherlands/1_prepare_feature_label_matrix.py
python -u scripts/scz_microarray_combined/1_prepare_feature_label_matrix.py
python -u scripts/scz_rnaseq_syn2759792/1_prepare_feature_label_matrix.py
python -u scripts/scz_rnaseq_syn4590909/1_prepare_feature_label_matrix.py
deactivate                                          # Deactivate virtual environment