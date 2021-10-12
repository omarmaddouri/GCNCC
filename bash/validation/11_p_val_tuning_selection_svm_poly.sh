#!/bin/bash
module load Python/3.6.6-intel-2018b
cd $SCRATCH/gcncc.bioinformatics                    # Go to project directory
source venv/bin/activate                            # Activate virtual environment (Command line should show environment name on left)
## Select the top features (i.e. activity scores)

#########################################################################################################################################################
#########################################################################################################################################################
##Radius=1
#########################################################################################################################################################
#########################################################################################################################################################
##brc_microarray_netherlands
python -u validation/6_select.py --dataset brc_microarray_netherlands --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 1 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/6_select.py --dataset brc_microarray_netherlands --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 1 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/6_select.py --dataset brc_microarray_netherlands --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 1 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/6_select.py --dataset brc_microarray_netherlands --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 1 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/6_select.py --dataset brc_microarray_netherlands --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 1 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.3      
#------------------------------------------------------------------------------------------------------------------------------------------------
##brc_microarray_usa
python -u validation/6_select.py --dataset brc_microarray_usa --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 1 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/6_select.py --dataset brc_microarray_usa --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 1 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/6_select.py --dataset brc_microarray_usa --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 1 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/6_select.py --dataset brc_microarray_usa --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 1 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/6_select.py --dataset brc_microarray_usa --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 1 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_microarray_combined
python -u validation/6_select.py --dataset scz_microarray_combined --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 1 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/6_select.py --dataset scz_microarray_combined --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 1 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/6_select.py --dataset scz_microarray_combined --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 1 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/6_select.py --dataset scz_microarray_combined --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 1 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/6_select.py --dataset scz_microarray_combined --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 1 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_rnaseq_syn2759792
python -u validation/6_select.py --dataset scz_rnaseq_syn2759792 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 1 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/6_select.py --dataset scz_rnaseq_syn2759792 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 1 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/6_select.py --dataset scz_rnaseq_syn2759792 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 1 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/6_select.py --dataset scz_rnaseq_syn2759792 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 1 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/6_select.py --dataset scz_rnaseq_syn2759792 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 1 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_rnaseq_syn4590909
python -u validation/6_select.py --dataset scz_rnaseq_syn4590909 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 1 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/6_select.py --dataset scz_rnaseq_syn4590909 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 1 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/6_select.py --dataset scz_rnaseq_syn4590909 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 1 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/6_select.py --dataset scz_rnaseq_syn4590909 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 1 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/6_select.py --dataset scz_rnaseq_syn4590909 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 1 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
#########################################################################################################################################################
#########################################################################################################################################################
##Radius=2
#########################################################################################################################################################
#########################################################################################################################################################
##brc_microarray_netherlands
python -u validation/6_select.py --dataset brc_microarray_netherlands --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 2 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/6_select.py --dataset brc_microarray_netherlands --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 2 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/6_select.py --dataset brc_microarray_netherlands --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 2 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/6_select.py --dataset brc_microarray_netherlands --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 2 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/6_select.py --dataset brc_microarray_netherlands --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 2 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.3      
#------------------------------------------------------------------------------------------------------------------------------------------------
##brc_microarray_usa
python -u validation/6_select.py --dataset brc_microarray_usa --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 2 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/6_select.py --dataset brc_microarray_usa --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 2 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/6_select.py --dataset brc_microarray_usa --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 2 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/6_select.py --dataset brc_microarray_usa --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 2 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/6_select.py --dataset brc_microarray_usa --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 2 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_microarray_combined
python -u validation/6_select.py --dataset scz_microarray_combined --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 2 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/6_select.py --dataset scz_microarray_combined --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 2 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/6_select.py --dataset scz_microarray_combined --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 2 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/6_select.py --dataset scz_microarray_combined --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 2 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/6_select.py --dataset scz_microarray_combined --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 2 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_rnaseq_syn2759792
python -u validation/6_select.py --dataset scz_rnaseq_syn2759792 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 2 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/6_select.py --dataset scz_rnaseq_syn2759792 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 2 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/6_select.py --dataset scz_rnaseq_syn2759792 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 2 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/6_select.py --dataset scz_rnaseq_syn2759792 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 2 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/6_select.py --dataset scz_rnaseq_syn2759792 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 2 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_rnaseq_syn4590909
python -u validation/6_select.py --dataset scz_rnaseq_syn4590909 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 2 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/6_select.py --dataset scz_rnaseq_syn4590909 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 2 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/6_select.py --dataset scz_rnaseq_syn4590909 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 2 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/6_select.py --dataset scz_rnaseq_syn4590909 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 2 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/6_select.py --dataset scz_rnaseq_syn4590909 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 2 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
#########################################################################################################################################################
#########################################################################################################################################################
##Radius=3
#########################################################################################################################################################
#########################################################################################################################################################
##brc_microarray_netherlands
python -u validation/6_select.py --dataset brc_microarray_netherlands --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 3 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/6_select.py --dataset brc_microarray_netherlands --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 3 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/6_select.py --dataset brc_microarray_netherlands --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 3 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/6_select.py --dataset brc_microarray_netherlands --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 3 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/6_select.py --dataset brc_microarray_netherlands --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 3 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.3      
#------------------------------------------------------------------------------------------------------------------------------------------------
##brc_microarray_usa
python -u validation/6_select.py --dataset brc_microarray_usa --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 3 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/6_select.py --dataset brc_microarray_usa --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 3 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/6_select.py --dataset brc_microarray_usa --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 3 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/6_select.py --dataset brc_microarray_usa --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 3 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/6_select.py --dataset brc_microarray_usa --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "free","metastasis" --distance_threshold 3 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_microarray_combined
python -u validation/6_select.py --dataset scz_microarray_combined --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 3 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/6_select.py --dataset scz_microarray_combined --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 3 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/6_select.py --dataset scz_microarray_combined --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 3 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/6_select.py --dataset scz_microarray_combined --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 3 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/6_select.py --dataset scz_microarray_combined --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 3 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_rnaseq_syn2759792
python -u validation/6_select.py --dataset scz_rnaseq_syn2759792 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 3 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/6_select.py --dataset scz_rnaseq_syn2759792 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 3 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/6_select.py --dataset scz_rnaseq_syn2759792 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 3 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/6_select.py --dataset scz_rnaseq_syn2759792 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 3 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/6_select.py --dataset scz_rnaseq_syn2759792 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 3 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_rnaseq_syn4590909
python -u validation/6_select.py --dataset scz_rnaseq_syn4590909 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 3 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/6_select.py --dataset scz_rnaseq_syn4590909 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 3 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/6_select.py --dataset scz_rnaseq_syn4590909 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 3 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/6_select.py --dataset scz_rnaseq_syn4590909 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 3 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/6_select.py --dataset scz_rnaseq_syn4590909 --classifier svm_poly --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --clusters clusters.txt --clusters_symbols clusters.symbols.txt --clusters_scores clusters.scores.txt\
        --labels "control","schizophrenia" --distance_threshold 3 --nb_features 5 --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
deactivate                                          # Deactivate virtual environment