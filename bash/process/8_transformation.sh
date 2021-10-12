#!/bin/bash
module load Python/3.6.6-intel-2018b
cd $SCRATCH/gcncc.bioinformatics                    # Go to project directory
source venv/bin/activate                            # Activate virtual environment (Command line should show environment name on left)

## Transform the data to activity scores

##Direct scoring: brc_microarray_netherlands
python -u process/5_transform.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method ig --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"

python -u process/5_transform.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method wgcna --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
        
python -u process/5_transform.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"        

python -u process/5_transform.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"

python -u process/5_transform.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
##Cross scoring: brc_microarray_netherlands
python -u process/5_transform.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method ig --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"

python -u process/5_transform.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method wgcna --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
        
python -u process/5_transform.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"        

python -u process/5_transform.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"

python -u process/5_transform.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
##Direct scoring: brc_microarray_usa
python -u process/5_transform.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method ig --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"

python -u process/5_transform.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method wgcna --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
        
python -u process/5_transform.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"        

python -u process/5_transform.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"

python -u process/5_transform.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
##Cross scoring: brc_microarray_usa
python -u process/5_transform.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method ig --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"

python -u process/5_transform.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method wgcna --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
        
python -u process/5_transform.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"        

python -u process/5_transform.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"

python -u process/5_transform.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
python -u process/5_transform.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis"
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
##Direct scoring: scz_microarray_combined
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method ig --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
        
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method wgcna --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"        
        
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"        

python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"

python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
##Cross scoring: scz_microarray_combined
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method ig --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
        
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method wgcna --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"        
        
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"        

python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"

python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
##Cross scoring: scz_microarray_combined
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method ig --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
        
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method wgcna --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"        
        
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"        

python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"

python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
##Direct scoring: scz_rnaseq_syn2759792
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method ig --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"

python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method wgcna --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"        
        
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"        

python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"

python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
##Cross scoring: scz_rnaseq_syn2759792
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method ig --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"

python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method wgcna --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"        
        
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"        

python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"

python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
##Cross scoring: scz_rnaseq_syn2759792
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method ig --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"

python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method wgcna --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"        
        
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"        

python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"

python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
##Direct scoring: scz_rnaseq_syn4590909
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method ig --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"

python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method wgcna --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
        
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"        

python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"

python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
##Cross scoring: scz_rnaseq_syn4590909
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method ig --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"

python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method wgcna --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
        
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"        

python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"

python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
##Cross scoring: scz_rnaseq_syn4590909
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method ig --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"

python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method wgcna --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
        
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method geometric_ap --embedding_method ge --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"        

python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method standard_ap --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"

python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method gcn --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method kernel_pca --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
python -u process/5_transform.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792 --full_data full_data.txt --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt\
        --clustering_method mcl --embedding_method vae --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia"
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
deactivate                                          # Deactivate virtual environment