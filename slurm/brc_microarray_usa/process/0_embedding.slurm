#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name="0_brc_microarray_usa_embeddings"       #Set the job name
#SBATCH --time=5:00:00              #Set the wall clock limit
#SBATCH --nodes=1                    #Request 1 node
#SBATCH --ntasks-per-node=8          #Request 8 tasks/cores per node
#SBATCH --mem=32G                  #Request 32GB per node
#SBATCH --output=0_brc_microarray_usa_embeddings.%j      #Send stdout/err to "out.[jobID]"
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

##Take gene expressions as embeddings
python -u process/1_embed_keep_ge.py --dataset brc_microarray_usa --embedding_method ge

##Generate embeddings by GCN
python -u process/1_embed_gcn.py --dataset brc_microarray_usa --embedding_method gcn --epochs 500 --embedding_size 64 --early_stopping 50

##Generate embeddings by kernel PCA
python -u process/1_embed_kernel_pca.py --dataset brc_microarray_usa --embedding_method kernel_pca --embedding_size 64

##Generate embeddings by VAE
python -u process/1_embed_vae.py --batch_size 100 --epochs 500 --kappa 1 --depth 1 --first_layer 128 --dataset brc_microarray_usa --embedding_method vae --embedding_size 64
deactivate                                          # Deactivate virtual environment