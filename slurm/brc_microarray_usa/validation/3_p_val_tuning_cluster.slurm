#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name="brc_microarray_usa_p_val_tuning_clustering"       #Set the job name
#SBATCH --time=20:00:00              #Set the wall clock limit
#SBATCH --nodes=1                    #Request 1 node
#SBATCH --ntasks-per-node=8          #Request 8 tasks/cores per node
#SBATCH --mem=48G                  #Request 48GB per node
#SBATCH --output=brc_microarray_usa_p_val_tuning_clustering.%j      #Send stdout/err to "out.[jobID]"
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


##Generate clusters
##Radius=1
##----------------------------------------------------------------------------------------------------------------------------
python -u validation/2_cluster_geometric_ap.py --dataset brc_microarray_usa --embeddings embeddings.txt --embedding_method gcn\
        --max_iter 300 --convergence_iter 15 --damp_factor 0.9\
        --distance_threshold 1 --similarity_metric "shortest_path"\
        --clustering_method geometric_ap --clusters_output clusters.txt --modularity "modularity.txt"\
        --hidden_layers 3 --embedding_size 64 --quantile 0.95 --desired_nb_clusters 500 --tolerance_nb_clusters 50 --p_val 0.01

python -u validation/2_cluster_geometric_ap.py --dataset brc_microarray_usa --embeddings embeddings.txt --embedding_method gcn\
        --max_iter 300 --convergence_iter 15 --damp_factor 0.9\
        --distance_threshold 1 --similarity_metric "shortest_path"\
        --clustering_method geometric_ap --clusters_output clusters.txt --modularity "modularity.txt"\
        --hidden_layers 3 --embedding_size 64 --quantile 0.95 --desired_nb_clusters 500 --tolerance_nb_clusters 50 --p_val 0.05

python -u validation/2_cluster_geometric_ap.py --dataset brc_microarray_usa --embeddings embeddings.txt --embedding_method gcn\
        --max_iter 300 --convergence_iter 15 --damp_factor 0.9\
        --distance_threshold 1 --similarity_metric "shortest_path"\
        --clustering_method geometric_ap --clusters_output clusters.txt --modularity "modularity.txt"\
        --hidden_layers 3 --embedding_size 64 --quantile 0.95 --desired_nb_clusters 500 --tolerance_nb_clusters 50 --p_val 0.1

python -u validation/2_cluster_geometric_ap.py --dataset brc_microarray_usa --embeddings embeddings.txt --embedding_method gcn\
        --max_iter 300 --convergence_iter 15 --damp_factor 0.9\
        --distance_threshold 1 --similarity_metric "shortest_path"\
        --clustering_method geometric_ap --clusters_output clusters.txt --modularity "modularity.txt"\
        --hidden_layers 3 --embedding_size 64 --quantile 0.95 --desired_nb_clusters 500 --tolerance_nb_clusters 50 --p_val 0.2

python -u validation/2_cluster_geometric_ap.py --dataset brc_microarray_usa --embeddings embeddings.txt --embedding_method gcn\
        --max_iter 300 --convergence_iter 15 --damp_factor 0.9\
        --distance_threshold 1 --similarity_metric "shortest_path"\
        --clustering_method geometric_ap --clusters_output clusters.txt --modularity "modularity.txt"\
        --hidden_layers 3 --embedding_size 64 --quantile 0.95 --desired_nb_clusters 500 --tolerance_nb_clusters 50 --p_val 0.3
##----------------------------------------------------------------------------------------------------------------------------
##Radius=2
##----------------------------------------------------------------------------------------------------------------------------
python -u validation/2_cluster_geometric_ap.py --dataset brc_microarray_usa --embeddings embeddings.txt --embedding_method gcn\
        --max_iter 300 --convergence_iter 15 --damp_factor 0.9\
        --distance_threshold 2 --similarity_metric "shortest_path"\
        --clustering_method geometric_ap --clusters_output clusters.txt --modularity "modularity.txt"\
        --hidden_layers 3 --embedding_size 64 --quantile 0.95 --desired_nb_clusters 500 --tolerance_nb_clusters 50 --p_val 0.01

python -u validation/2_cluster_geometric_ap.py --dataset brc_microarray_usa --embeddings embeddings.txt --embedding_method gcn\
        --max_iter 300 --convergence_iter 15 --damp_factor 0.9\
        --distance_threshold 2 --similarity_metric "shortest_path"\
        --clustering_method geometric_ap --clusters_output clusters.txt --modularity "modularity.txt"\
        --hidden_layers 3 --embedding_size 64 --quantile 0.95 --desired_nb_clusters 500 --tolerance_nb_clusters 50 --p_val 0.05

python -u validation/2_cluster_geometric_ap.py --dataset brc_microarray_usa --embeddings embeddings.txt --embedding_method gcn\
        --max_iter 300 --convergence_iter 15 --damp_factor 0.9\
        --distance_threshold 2 --similarity_metric "shortest_path"\
        --clustering_method geometric_ap --clusters_output clusters.txt --modularity "modularity.txt"\
        --hidden_layers 3 --embedding_size 64 --quantile 0.95 --desired_nb_clusters 500 --tolerance_nb_clusters 50 --p_val 0.1

python -u validation/2_cluster_geometric_ap.py --dataset brc_microarray_usa --embeddings embeddings.txt --embedding_method gcn\
        --max_iter 300 --convergence_iter 15 --damp_factor 0.9\
        --distance_threshold 2 --similarity_metric "shortest_path"\
        --clustering_method geometric_ap --clusters_output clusters.txt --modularity "modularity.txt"\
        --hidden_layers 3 --embedding_size 64 --quantile 0.95 --desired_nb_clusters 500 --tolerance_nb_clusters 50 --p_val 0.2

python -u validation/2_cluster_geometric_ap.py --dataset brc_microarray_usa --embeddings embeddings.txt --embedding_method gcn\
        --max_iter 300 --convergence_iter 15 --damp_factor 0.9\
        --distance_threshold 2 --similarity_metric "shortest_path"\
        --clustering_method geometric_ap --clusters_output clusters.txt --modularity "modularity.txt"\
        --hidden_layers 3 --embedding_size 64 --quantile 0.95 --desired_nb_clusters 500 --tolerance_nb_clusters 50 --p_val 0.3
##----------------------------------------------------------------------------------------------------------------------------
##Radius=3
##----------------------------------------------------------------------------------------------------------------------------
python -u validation/2_cluster_geometric_ap.py --dataset brc_microarray_usa --embeddings embeddings.txt --embedding_method gcn\
        --max_iter 300 --convergence_iter 15 --damp_factor 0.9\
        --distance_threshold 3 --similarity_metric "shortest_path"\
        --clustering_method geometric_ap --clusters_output clusters.txt --modularity "modularity.txt"\
        --hidden_layers 3 --embedding_size 64 --quantile 0.95 --desired_nb_clusters 500 --tolerance_nb_clusters 50 --p_val 0.01

python -u validation/2_cluster_geometric_ap.py --dataset brc_microarray_usa --embeddings embeddings.txt --embedding_method gcn\
        --max_iter 300 --convergence_iter 15 --damp_factor 0.9\
        --distance_threshold 3 --similarity_metric "shortest_path"\
        --clustering_method geometric_ap --clusters_output clusters.txt --modularity "modularity.txt"\
        --hidden_layers 3 --embedding_size 64 --quantile 0.95 --desired_nb_clusters 500 --tolerance_nb_clusters 50 --p_val 0.05

python -u validation/2_cluster_geometric_ap.py --dataset brc_microarray_usa --embeddings embeddings.txt --embedding_method gcn\
        --max_iter 300 --convergence_iter 15 --damp_factor 0.9\
        --distance_threshold 3 --similarity_metric "shortest_path"\
        --clustering_method geometric_ap --clusters_output clusters.txt --modularity "modularity.txt"\
        --hidden_layers 3 --embedding_size 64 --quantile 0.95 --desired_nb_clusters 500 --tolerance_nb_clusters 50 --p_val 0.1

python -u validation/2_cluster_geometric_ap.py --dataset brc_microarray_usa --embeddings embeddings.txt --embedding_method gcn\
        --max_iter 300 --convergence_iter 15 --damp_factor 0.9\
        --distance_threshold 3 --similarity_metric "shortest_path"\
        --clustering_method geometric_ap --clusters_output clusters.txt --modularity "modularity.txt"\
        --hidden_layers 3 --embedding_size 64 --quantile 0.95 --desired_nb_clusters 500 --tolerance_nb_clusters 50 --p_val 0.2

python -u validation/2_cluster_geometric_ap.py --dataset brc_microarray_usa --embeddings embeddings.txt --embedding_method gcn\
        --max_iter 300 --convergence_iter 15 --damp_factor 0.9\
        --distance_threshold 3 --similarity_metric "shortest_path"\
        --clustering_method geometric_ap --clusters_output clusters.txt --modularity "modularity.txt"\
        --hidden_layers 3 --embedding_size 64 --quantile 0.95 --desired_nb_clusters 500 --tolerance_nb_clusters 50 --p_val 0.3
##----------------------------------------------------------------------------------------------------------------------------
deactivate