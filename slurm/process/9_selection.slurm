#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name="9_GCNCC_feature_selection"       #Set the job name
#SBATCH --time=02:00:00              #Set the wall clock limit
#SBATCH --nodes=1                    #Request 1 node
#SBATCH --ntasks-per-node=12          #Request 12 tasks/cores per node
#SBATCH --mem=4G                  #Request 4GB per node
#SBATCH --output=9_GCNCC_feature_selection.%j      #Send stdout/err to "out.[jobID]"
##SBATCH --gres=gpu:1                 #Request 1 GPU per node can be 1 or 2
##SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --account=xxxxxxxxxxxx             #Set billing account to xxxxxxxxxxxx
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=omar.maddouri@tamu.edu    #Send all emails to email_address

#Load modules
module load Python/3.6.6-intel-2018b
cd $SCRATCH/gcncc.bioinformatics                    # Go to project directory
source venv/bin/activate                            # Activate virtual environment (Command line should show environment name on left)
## Generate clusters symbols for geometric_ap

##brc_microarray_netherlands
##Direct validation
python -u process/6_select.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method ig --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4

python -u process/6_select.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method wgcna --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
        
python -u process/6_select.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4        
python -u process/6_select.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
python -u process/6_select.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
python -u process/6_select.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4        

python -u process/6_select.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method standard_ap --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
python -u process/6_select.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method standard_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
python -u process/6_select.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method standard_ap --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4

python -u process/6_select.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method mcl --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
python -u process/6_select.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method mcl --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
python -u process/6_select.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_netherlands\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method mcl --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
#------------------------------------------------------------------------------------------------------------------------------------------------
##Cross dataset validation with feature selection based on the validation set from the testing dataset
python -u process/6_select.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method ig --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4

python -u process/6_select.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method wgcna --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4

python -u process/6_select.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4        
python -u process/6_select.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
python -u process/6_select.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
python -u process/6_select.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4        

python -u process/6_select.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method standard_ap --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
python -u process/6_select.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method standard_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
python -u process/6_select.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method standard_ap --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4

python -u process/6_select.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method mcl --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
python -u process/6_select.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method mcl --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
python -u process/6_select.py --train_dataset brc_microarray_netherlands --test_dataset brc_microarray_usa\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method mcl --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
#------------------------------------------------------------------------------------------------------------------------------------------------
#################################################################################################################################################
#################################################################################################################################################
##brc_microarray_usa
##Direct validation
python -u process/6_select.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method ig --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4

python -u process/6_select.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method wgcna --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4

python -u process/6_select.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4        
python -u process/6_select.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
python -u process/6_select.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
python -u process/6_select.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4        

python -u process/6_select.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method standard_ap --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
python -u process/6_select.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method standard_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
python -u process/6_select.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method standard_ap --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4

python -u process/6_select.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method mcl --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
python -u process/6_select.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method mcl --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
python -u process/6_select.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_usa\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method mcl --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
#------------------------------------------------------------------------------------------------------------------------------------------------
##Cross dataset validation with feature selection based on the validation set from the testing dataset
python -u process/6_select.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method ig --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4

python -u process/6_select.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method wgcna --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4

python -u process/6_select.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4        
python -u process/6_select.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
python -u process/6_select.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
python -u process/6_select.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4        

python -u process/6_select.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method standard_ap --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
python -u process/6_select.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method standard_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
python -u process/6_select.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method standard_ap --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4

python -u process/6_select.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method mcl --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
python -u process/6_select.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method mcl --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
python -u process/6_select.py --train_dataset brc_microarray_usa --test_dataset brc_microarray_netherlands\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method mcl --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "free","metastasis" --nb_features 4
#------------------------------------------------------------------------------------------------------------------------------------------------
#################################################################################################################################################
#################################################################################################################################################
##scz_microarray_combined
##Direct validation
python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method ig --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4

python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method wgcna --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4

python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4        
python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4        

python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method standard_ap --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method standard_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method standard_ap --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4

python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method mcl --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method mcl --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_microarray_combined\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method mcl --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
#------------------------------------------------------------------------------------------------------------------------------------------------
##Cross dataset validation with feature selection based on the validation set from the testing dataset
python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method ig --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4

python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method wgcna --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4

python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4        
python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4        

python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method standard_ap --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method standard_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method standard_ap --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4

python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method mcl --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method mcl --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method mcl --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
#------------------------------------------------------------------------------------------------------------------------------------------------
##Cross dataset validation with feature selection based on the validation set from the testing dataset
python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method ig --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4

python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method wgcna --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4        

python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4        
python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4        

python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method standard_ap --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method standard_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method standard_ap --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4

python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method mcl --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method mcl --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_microarray_combined --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method mcl --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
#------------------------------------------------------------------------------------------------------------------------------------------------
#################################################################################################################################################
#################################################################################################################################################
##scz_rnaseq_syn2759792
##Direct validation
python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method ig --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4

python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method wgcna --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4

python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4        
python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4        

python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method standard_ap --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method standard_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method standard_ap --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4

python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method mcl --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method mcl --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method mcl --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
#------------------------------------------------------------------------------------------------------------------------------------------------
##Cross dataset validation with feature selection based on the validation set from the testing dataset
python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method ig --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4

python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method wgcna --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4        

python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4        
python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4        

python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method standard_ap --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method standard_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method standard_ap --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4

python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method mcl --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method mcl --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_microarray_combined\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method mcl --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
#------------------------------------------------------------------------------------------------------------------------------------------------
##Cross dataset validation with feature selection based on the validation set from the testing dataset
python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method ig --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4

python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method wgcna --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4        

python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4        
python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
        
python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method standard_ap --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method standard_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method standard_ap --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4

python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method mcl --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method mcl --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn2759792 --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method mcl --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
#------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------
#################################################################################################################################################
#################################################################################################################################################
##scz_rnaseq_syn4590909
##Direct validation
python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method ig --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4

python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method wgcna --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4        

python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4        
python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method geometric_ap --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4        

python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method standard_ap --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method standard_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method standard_ap --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4

python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method mcl --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method mcl --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn4590909\
        --training_activity activity.training.txt --validation_activity activity.validation.txt --testing_activity activity.testing.txt\
        --clustering_method mcl --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
#------------------------------------------------------------------------------------------------------------------------------------------------
##Cross dataset validation with feature selection based on the validation set from the testing dataset
python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method ig --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4

python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method wgcna --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4        

python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4        
python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4        

python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method standard_ap --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method standard_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method standard_ap --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4

python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method mcl --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method mcl --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_microarray_combined\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method mcl --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
#------------------------------------------------------------------------------------------------------------------------------------------------
##Cross dataset validation with feature selection based on the validation set from the testing dataset
python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method ig --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4

python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method wgcna --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4        

python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method ge --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4        
python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method geometric_ap --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
        
python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method standard_ap --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method standard_ap --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method standard_ap --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4

python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method mcl --embedding_method gcn --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method mcl --embedding_method kernel_pca --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
python -u process/6_select.py --train_dataset scz_rnaseq_syn4590909 --test_dataset scz_rnaseq_syn2759792\
        --training_activity activity.full.txt --validation_activity activity.validation.txt --testing_activity activity.extended_testing.txt\
        --clustering_method mcl --embedding_method vae --clusters clusters.txt --clusters_symbols clusters.symbols.txt\
        --clusters_scores clusters.scores.txt --labels "control","schizophrenia" --nb_features 4
#------------------------------------------------------------------------------------------------------------------------------------------------
deactivate                                          # Deactivate virtual environment