#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name="1_brc_microarray_netherlands_clustering"       #Set the job name
#SBATCH --time=20:00:00              #Set the wall clock limit
#SBATCH --nodes=1                    #Request 1 node
#SBATCH --ntasks-per-node=8          #Request 8 tasks/cores per node
#SBATCH --mem=48G                  #Request 48GB per node
#SBATCH --output=1_brc_microarray_netherlands_clustering.%j      #Send stdout/err to "out.[jobID]"
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
module load  R_tamu/3.6.3-foss-2020a
export R_LIBS_USER=${SCRATCH}/myRlibs


##Generate clusters

##Clustering method: ig
##----------------------------------------------------------------------------------------------------------------------------
python -u process/2_cluster_ig.py --dataset brc_microarray_netherlands --embeddings embeddings.txt --embedding_method ge\
        --clustering_method ig --clusters_output clusters.txt
        
##Clustering method: geometric AP
##----------------------------------------------------------------------------------------------------------------------------
python -u process/2_cluster_geometric_ap.py --dataset brc_microarray_netherlands --embeddings embeddings.txt --embedding_method ge\
        --distance_threshold 2 --max_iter 200 --convergence_iter 15 --damp_factor 0.9 --clustering_method geometric_ap --clusters_output clusters.txt\
        --quantile 0.2 --desired_nb_clusters 500 --tolerance_nb_clusters 100
        
python -u process/2_cluster_geometric_ap.py --dataset brc_microarray_netherlands --embeddings embeddings.txt --embedding_method gcn\
        --distance_threshold 2 --max_iter 200 --convergence_iter 15 --damp_factor 0.9 --clustering_method geometric_ap --clusters_output clusters.txt\
        --quantile 0.2 --desired_nb_clusters 500 --tolerance_nb_clusters 100
        
python -u process/2_cluster_geometric_ap.py --dataset brc_microarray_netherlands --embeddings embeddings.txt --embedding_method kernel_pca\
        --distance_threshold 2 --max_iter 200 --convergence_iter 15 --damp_factor 0.9 --clustering_method geometric_ap --clusters_output clusters.txt\
        --quantile 0.2 --desired_nb_clusters 500 --tolerance_nb_clusters 100

python -u process/2_cluster_geometric_ap.py --dataset brc_microarray_netherlands --embeddings embeddings.txt --embedding_method vae\
        --distance_threshold 2 --max_iter 200 --convergence_iter 15 --damp_factor 0.9 --clustering_method geometric_ap --clusters_output clusters.txt\
        --quantile 0.2 --desired_nb_clusters 500 --tolerance_nb_clusters 100      
##----------------------------------------------------------------------------------------------------------------------------

##Clustering method: standard AP
##----------------------------------------------------------------------------------------------------------------------------
python -u process/2_cluster_standard_ap.py --dataset brc_microarray_netherlands --embeddings embeddings.txt --embedding_method gcn\
        --max_iter 200 --convergence_iter 15 --damp_factor 0.9 --clustering_method standard_ap --clusters_output clusters.txt\
        --quantile 0.2 --desired_nb_clusters 500 --tolerance_nb_clusters 100
        
python -u process/2_cluster_standard_ap.py --dataset brc_microarray_netherlands --embeddings embeddings.txt --embedding_method kernel_pca\
        --max_iter 200 --convergence_iter 15 --damp_factor 0.9 --clustering_method standard_ap --clusters_output clusters.txt\
        --quantile 0.2 --desired_nb_clusters 500 --tolerance_nb_clusters 100

python -u process/2_cluster_standard_ap.py --dataset brc_microarray_netherlands --embeddings embeddings.txt --embedding_method vae\
        --max_iter 200 --convergence_iter 15 --damp_factor 0.9 --clustering_method standard_ap --clusters_output clusters.txt\
        --quantile 0.01 --desired_nb_clusters 1000 --tolerance_nb_clusters 500        
##----------------------------------------------------------------------------------------------------------------------------

##Clustering method: MCL
##----------------------------------------------------------------------------------------------------------------------------
python -u process/2_cluster_mcl.py --dataset brc_microarray_netherlands --embeddings embeddings.txt --embedding_method gcn\
        --clustering_method mcl --clusters_output clusters.txt
        
python -u process/2_cluster_mcl.py --dataset brc_microarray_netherlands --embeddings embeddings.txt --embedding_method kernel_pca\
        --clustering_method mcl --clusters_output clusters.txt

python -u process/2_cluster_mcl.py --dataset brc_microarray_netherlands --embeddings embeddings.txt --embedding_method vae\
        --clustering_method mcl --clusters_output clusters.txt
##----------------------------------------------------------------------------------------------------------------------------
deactivate