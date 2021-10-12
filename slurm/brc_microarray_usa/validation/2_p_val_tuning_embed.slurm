#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name="brc_microarray_usa_p_val_tuning_embedding"       #Set the job name
#SBATCH --time=10:00:00              #Set the wall clock limit
#SBATCH --nodes=1                    #Request 1 node
#SBATCH --ntasks-per-node=8          #Request 8 tasks/cores per node
#SBATCH --mem=32G                  #Request 32GB per node
#SBATCH --output=brc_microarray_usa_p_val_tuning_embedding.%j      #Send stdout/err to "out.[jobID]"
##SBATCH --gres=gpu:1                 #Request 1 GPU per node can be 1 or 2
##SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --account=xxxxxxxxxxxx             #Set billing account to xxxxxxxxxxxx
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=omar.maddouri@tamu.edu    #Send all emails to email_address

#Load modules
cd $SCRATCH/gcncc.bioinformatics
module purge
module load Python/3.6.6-intel-2018b
source venv/bin/activate

##Generate node embeddings
##----------------------------------------------------------------------------------------------------------------------------
python -u  validation/1_embed_gcn_3_hidden_layers.py --dataset brc_microarray_usa --p_value_threshold 0.01 --embedding_size 64
python -u  validation/1_embed_gcn_3_hidden_layers.py --dataset brc_microarray_usa --p_value_threshold 0.05 --embedding_size 64
python -u  validation/1_embed_gcn_3_hidden_layers.py --dataset brc_microarray_usa --p_value_threshold 0.1 --embedding_size 64
python -u  validation/1_embed_gcn_3_hidden_layers.py --dataset brc_microarray_usa --p_value_threshold 0.2 --embedding_size 64
python -u  validation/1_embed_gcn_3_hidden_layers.py --dataset brc_microarray_usa --p_value_threshold 0.3 --embedding_size 64
##----------------------------------------------------------------------------------------------------------------------------

deactivate