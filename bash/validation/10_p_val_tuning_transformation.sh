#!/bin/bash
module load Python/3.6.6-intel-2018b
cd $SCRATCH/gcncc.bioinformatics                    # Go to project directory
source venv/bin/activate                            # Activate virtual environment (Command line should show environment name on left)
## Transform the data to activity scores

#########################################################################################################################################################
#########################################################################################################################################################
##Radius=1
#########################################################################################################################################################
#########################################################################################################################################################
##brc_microarray_netherlands
python -u validation/5_transform.py --dataset brc_microarray_netherlands --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 1 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/5_transform.py --dataset brc_microarray_netherlands --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 1 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/5_transform.py --dataset brc_microarray_netherlands --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 1 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/5_transform.py --dataset brc_microarray_netherlands --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 1 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/5_transform.py --dataset brc_microarray_netherlands --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 1 --hidden_layers 3 --embedding_size 64 --p_val 0.3      
    
#------------------------------------------------------------------------------------------------------------------------------------------------
##brc_microarray_usa
python -u validation/5_transform.py --dataset brc_microarray_usa --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 1 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/5_transform.py --dataset brc_microarray_usa --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 1 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/5_transform.py --dataset brc_microarray_usa --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 1 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/5_transform.py --dataset brc_microarray_usa --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 1 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/5_transform.py --dataset brc_microarray_usa --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 1 --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_microarray_combined
python -u validation/5_transform.py --dataset scz_microarray_combined --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 1 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/5_transform.py --dataset scz_microarray_combined --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 1 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/5_transform.py --dataset scz_microarray_combined --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 1 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/5_transform.py --dataset scz_microarray_combined --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 1 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/5_transform.py --dataset scz_microarray_combined --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 1 --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_rnaseq_syn2759792
python -u validation/5_transform.py --dataset scz_rnaseq_syn2759792 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 1 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/5_transform.py --dataset scz_rnaseq_syn2759792 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 1 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/5_transform.py --dataset scz_rnaseq_syn2759792 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 1 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/5_transform.py --dataset scz_rnaseq_syn2759792 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 1 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/5_transform.py --dataset scz_rnaseq_syn2759792 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 1 --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_rnaseq_syn4590909
python -u validation/5_transform.py --dataset scz_rnaseq_syn4590909 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 1 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/5_transform.py --dataset scz_rnaseq_syn4590909 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 1 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/5_transform.py --dataset scz_rnaseq_syn4590909 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 1 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/5_transform.py --dataset scz_rnaseq_syn4590909 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 1 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/5_transform.py --dataset scz_rnaseq_syn4590909 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 1 --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
#########################################################################################################################################################
#########################################################################################################################################################
##Radius=2
#########################################################################################################################################################
#########################################################################################################################################################
##brc_microarray_netherlands
python -u validation/5_transform.py --dataset brc_microarray_netherlands --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 2 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/5_transform.py --dataset brc_microarray_netherlands --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 2 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/5_transform.py --dataset brc_microarray_netherlands --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 2 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/5_transform.py --dataset brc_microarray_netherlands --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 2 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/5_transform.py --dataset brc_microarray_netherlands --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 2 --hidden_layers 3 --embedding_size 64 --p_val 0.3      
    
#------------------------------------------------------------------------------------------------------------------------------------------------
##brc_microarray_usa
python -u validation/5_transform.py --dataset brc_microarray_usa --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 2 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/5_transform.py --dataset brc_microarray_usa --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 2 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/5_transform.py --dataset brc_microarray_usa --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 2 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/5_transform.py --dataset brc_microarray_usa --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 2 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/5_transform.py --dataset brc_microarray_usa --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 2 --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_microarray_combined
python -u validation/5_transform.py --dataset scz_microarray_combined --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 2 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/5_transform.py --dataset scz_microarray_combined --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 2 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/5_transform.py --dataset scz_microarray_combined --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 2 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/5_transform.py --dataset scz_microarray_combined --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 2 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/5_transform.py --dataset scz_microarray_combined --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 2 --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_rnaseq_syn2759792
python -u validation/5_transform.py --dataset scz_rnaseq_syn2759792 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 2 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/5_transform.py --dataset scz_rnaseq_syn2759792 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 2 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/5_transform.py --dataset scz_rnaseq_syn2759792 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 2 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/5_transform.py --dataset scz_rnaseq_syn2759792 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 2 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/5_transform.py --dataset scz_rnaseq_syn2759792 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 2 --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_rnaseq_syn4590909
python -u validation/5_transform.py --dataset scz_rnaseq_syn4590909 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 2 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/5_transform.py --dataset scz_rnaseq_syn4590909 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 2 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/5_transform.py --dataset scz_rnaseq_syn4590909 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 2 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/5_transform.py --dataset scz_rnaseq_syn4590909 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 2 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/5_transform.py --dataset scz_rnaseq_syn4590909 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 2 --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
#########################################################################################################################################################
#########################################################################################################################################################
##Radius=3
#########################################################################################################################################################
#########################################################################################################################################################
##brc_microarray_netherlands
python -u validation/5_transform.py --dataset brc_microarray_netherlands --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 3 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/5_transform.py --dataset brc_microarray_netherlands --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 3 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/5_transform.py --dataset brc_microarray_netherlands --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 3 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/5_transform.py --dataset brc_microarray_netherlands --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 3 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/5_transform.py --dataset brc_microarray_netherlands --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 3 --hidden_layers 3 --embedding_size 64 --p_val 0.3      
    
#------------------------------------------------------------------------------------------------------------------------------------------------
##brc_microarray_usa
python -u validation/5_transform.py --dataset brc_microarray_usa --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 3 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/5_transform.py --dataset brc_microarray_usa --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 3 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/5_transform.py --dataset brc_microarray_usa --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 3 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/5_transform.py --dataset brc_microarray_usa --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 3 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/5_transform.py --dataset brc_microarray_usa --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "free","metastasis" --distance_threshold 3 --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_microarray_combined
python -u validation/5_transform.py --dataset scz_microarray_combined --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 3 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/5_transform.py --dataset scz_microarray_combined --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 3 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/5_transform.py --dataset scz_microarray_combined --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 3 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/5_transform.py --dataset scz_microarray_combined --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 3 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/5_transform.py --dataset scz_microarray_combined --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 3 --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_rnaseq_syn2759792
python -u validation/5_transform.py --dataset scz_rnaseq_syn2759792 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 3 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/5_transform.py --dataset scz_rnaseq_syn2759792 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 3 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/5_transform.py --dataset scz_rnaseq_syn2759792 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 3 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/5_transform.py --dataset scz_rnaseq_syn2759792 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 3 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/5_transform.py --dataset scz_rnaseq_syn2759792 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 3 --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_rnaseq_syn4590909
python -u validation/5_transform.py --dataset scz_rnaseq_syn4590909 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 3 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/5_transform.py --dataset scz_rnaseq_syn4590909 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 3 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/5_transform.py --dataset scz_rnaseq_syn4590909 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 3 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/5_transform.py --dataset scz_rnaseq_syn4590909 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 3 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/5_transform.py --dataset scz_rnaseq_syn4590909 --training_data training_data.txt --testing_data testing_data.txt --validation_data validation_data.txt --clustering_method geometric_ap\
        --clusters clusters.txt --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --distance_threshold 3 --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
deactivate                                          # Deactivate virtual environment