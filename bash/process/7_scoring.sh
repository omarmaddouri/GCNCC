#!/bin/bash
module load Python/3.6.6-intel-2018b
cd $SCRATCH/gcncc.bioinformatics                    # Go to project directory
source venv/bin/activate                            # Activate virtual environment (Command line should show environment name on left)

## Score clusters

##Direct scoring: brc_microarray_netherlands
python -u process/4_score.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands --clustering_method ig --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"

python -u process/4_score.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands --clustering_method wgcna --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"

python -u process/4_score.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands --clustering_method geometric_ap --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"        
python -u process/4_score.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands --clustering_method geometric_ap --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands --clustering_method geometric_ap --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands --clustering_method geometric_ap --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"

python -u process/4_score.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands --clustering_method standard_ap --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands --clustering_method standard_ap --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands --clustering_method standard_ap --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
        
python -u process/4_score.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands --clustering_method mcl --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands --clustering_method mcl --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands --clustering_method mcl --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
#------------------------------------------------------------------------------------------------------------------------------------------------
##Cross scoring: brc_microarray_netherlands
python -u process/4_score.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa --clustering_method ig --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"

python -u process/4_score.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa --clustering_method wgcna --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
        
python -u process/4_score.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa --clustering_method geometric_ap --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa --clustering_method geometric_ap --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa --clustering_method geometric_ap --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa --clustering_method geometric_ap --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"

python -u process/4_score.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa --clustering_method standard_ap --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa --clustering_method standard_ap --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa --clustering_method standard_ap --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
        
python -u process/4_score.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa --clustering_method mcl --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa --clustering_method mcl --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa --clustering_method mcl --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
#------------------------------------------------------------------------------------------------------------------------------------------------
##Direct scoring: brc_microarray_usa
python -u process/4_score.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa --clustering_method ig --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"

python -u process/4_score.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa --clustering_method wgcna --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
        
python -u process/4_score.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa --clustering_method geometric_ap --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa --clustering_method geometric_ap --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa --clustering_method geometric_ap --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa --clustering_method geometric_ap --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"

python -u process/4_score.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa --clustering_method standard_ap --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa --clustering_method standard_ap --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa --clustering_method standard_ap --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
        
python -u process/4_score.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa --clustering_method mcl --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa --clustering_method mcl --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa --clustering_method mcl --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
#------------------------------------------------------------------------------------------------------------------------------------------------
##Cross scoring: brc_microarray_usa
python -u process/4_score.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands --clustering_method ig --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"

python -u process/4_score.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands --clustering_method wgcna --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
        
python -u process/4_score.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands --clustering_method geometric_ap --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands --clustering_method geometric_ap --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands --clustering_method geometric_ap --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands --clustering_method geometric_ap --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"

python -u process/4_score.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands --clustering_method standard_ap --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands --clustering_method standard_ap --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands --clustering_method standard_ap --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
        
python -u process/4_score.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands --clustering_method mcl --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands --clustering_method mcl --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
python -u process/4_score.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands --clustering_method mcl --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "free","metastasis"
#------------------------------------------------------------------------------------------------------------------------------------------------
##Direct scoring: scz_microarray_combined
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined --clustering_method ig --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"

python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined --clustering_method wgcna --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
        
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined --clustering_method geometric_ap --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined --clustering_method geometric_ap --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined --clustering_method geometric_ap --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined --clustering_method geometric_ap --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"

python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined --clustering_method standard_ap --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined --clustering_method standard_ap --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined --clustering_method standard_ap --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
        
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined --clustering_method mcl --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined --clustering_method mcl --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined --clustering_method mcl --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
#------------------------------------------------------------------------------------------------------------------------------------------------
##Cross scoring: scz_microarray_combined
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792 --clustering_method ig --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"

python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792 --clustering_method wgcna --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
        
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"

python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792 --clustering_method standard_ap --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792 --clustering_method standard_ap --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792 --clustering_method standard_ap --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
        
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792 --clustering_method mcl --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792 --clustering_method mcl --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792 --clustering_method mcl --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
#------------------------------------------------------------------------------------------------------------------------------------------------
##Cross scoring: scz_microarray_combined
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909 --clustering_method ig --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"

python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909 --clustering_method wgcna --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
        
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"

python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909 --clustering_method standard_ap --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909 --clustering_method standard_ap --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909 --clustering_method standard_ap --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
        
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909 --clustering_method mcl --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909 --clustering_method mcl --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909 --clustering_method mcl --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
#------------------------------------------------------------------------------------------------------------------------------------------------
##Direct scoring: scz_rnaseq_syn2759792
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792 --clustering_method ig --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"

python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792 --clustering_method wgcna --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
        
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"

python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792 --clustering_method standard_ap --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792 --clustering_method standard_ap --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792 --clustering_method standard_ap --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
        
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792 --clustering_method mcl --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792 --clustering_method mcl --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792 --clustering_method mcl --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
#------------------------------------------------------------------------------------------------------------------------------------------------
##Cross scoring: scz_rnaseq_syn2759792
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined --clustering_method ig --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"

python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined --clustering_method wgcna --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
        
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined --clustering_method geometric_ap --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined --clustering_method geometric_ap --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined --clustering_method geometric_ap --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined --clustering_method geometric_ap --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"

python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined --clustering_method standard_ap --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined --clustering_method standard_ap --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined --clustering_method standard_ap --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
        
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined --clustering_method mcl --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined --clustering_method mcl --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined --clustering_method mcl --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
#------------------------------------------------------------------------------------------------------------------------------------------------
##Cross scoring: scz_rnaseq_syn2759792
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909 --clustering_method ig --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"

python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909 --clustering_method wgcna --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
        
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"

python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909 --clustering_method standard_ap --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909 --clustering_method standard_ap --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909 --clustering_method standard_ap --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
        
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909 --clustering_method mcl --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909 --clustering_method mcl --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909 --clustering_method mcl --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
#------------------------------------------------------------------------------------------------------------------------------------------------
##Direct scoring: scz_rnaseq_syn4590909
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909 --clustering_method ig --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"

python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909 --clustering_method wgcna --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
        
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"

python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909 --clustering_method standard_ap --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909 --clustering_method standard_ap --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909 --clustering_method standard_ap --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
        
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909 --clustering_method mcl --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909 --clustering_method mcl --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909 --clustering_method mcl --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
#------------------------------------------------------------------------------------------------------------------------------------------------
##Cross scoring: scz_rnaseq_syn4590909
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined --clustering_method ig --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"

python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined --clustering_method wgcna --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
        
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined --clustering_method geometric_ap --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined --clustering_method geometric_ap --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined --clustering_method geometric_ap --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined --clustering_method geometric_ap --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"

python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined --clustering_method standard_ap --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined --clustering_method standard_ap --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined --clustering_method standard_ap --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
        
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined --clustering_method mcl --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined --clustering_method mcl --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined --clustering_method mcl --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
#------------------------------------------------------------------------------------------------------------------------------------------------
##Cross scoring: scz_rnaseq_syn4590909
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792 --clustering_method ig --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"

python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792 --clustering_method wgcna --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
        
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --embedding_method ge\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"

python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792 --clustering_method standard_ap --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792 --clustering_method standard_ap --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792 --clustering_method standard_ap --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
        
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792 --clustering_method mcl --embedding_method gcn\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792 --clustering_method mcl --embedding_method kernel_pca\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
python -u process/4_score.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792 --clustering_method mcl --embedding_method vae\
        --scoring_data full_data.txt --clusters clusters.txt --labels "control","schizophrenia"
#------------------------------------------------------------------------------------------------------------------------------------------------

deactivate                                          # Deactivate virtual environment