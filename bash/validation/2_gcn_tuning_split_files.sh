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
##Split datasets for  given p_values
python -u validation/0_split_data.py --dataset brc_microarray_usa --p_value_threshold 0.1 --test_split_size 0.2 --val_split_size 0.2 --labels "free","metastasis"
python -u validation/0_split_data.py --dataset brc_microarray_netherlands --p_value_threshold 0.1 --test_split_size 0.2 --val_split_size 0.2  --labels "free","metastasis"
python -u validation/0_split_data.py --dataset scz_microarray_combined --p_value_threshold 0.1 --test_split_size 0.2 --val_split_size 0.2  --labels "control","schizophrenia"
python -u validation/0_split_data.py --dataset scz_rnaseq_syn2759792 --p_value_threshold 0.1 --test_split_size 0.2 --val_split_size 0.2  --labels "control","schizophrenia"
python -u validation/0_split_data.py --dataset scz_rnaseq_syn4590909 --p_value_threshold 0.1 --test_split_size 0.2 --val_split_size 0.2  --labels "control","schizophrenia"
deactivate                                          # Deactivate virtual environment