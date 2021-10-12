#!/bin/bash
module load Python/3.6.6-intel-2018b
cd $SCRATCH/gcncc.bioinformatics                    # Go to project directory
source venv/bin/activate                            # Activate virtual environment (Command line should show environment name on left)
## Generate clusters symbols for geometric_ap
#########################################################################################################################################################
#########################################################################################################################################################
##Radius=1
#########################################################################################################################################################
#########################################################################################################################################################
##brc_microarray_netherlands
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 1 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 1 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 1 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 1 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 1 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##brc_microarray_usa
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 1 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 1 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 1 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 1 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 1 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_microarray_combined
python -u validation/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 1 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 1 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 1 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 1 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 1 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_rnaseq_syn2759792
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 1 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 1 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 1 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 1 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 1 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_rnaseq_syn4590909
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 1 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 1 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 1 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 1 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 1 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
#########################################################################################################################################################
#########################################################################################################################################################
##Radius=2
#########################################################################################################################################################
#########################################################################################################################################################
##brc_microarray_netherlands
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 2 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 2 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 2 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 2 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 2 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##brc_microarray_usa
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 2 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 2 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 2 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 2 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 2 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_microarray_combined
python -u validation/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 2 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 2 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 2 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 2 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 2 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_rnaseq_syn2759792
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 2 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 2 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 2 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 2 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 2 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_rnaseq_syn4590909
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 2 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 2 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 2 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 2 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 2 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
#########################################################################################################################################################
#########################################################################################################################################################
##Radius=3
#########################################################################################################################################################
#########################################################################################################################################################
##brc_microarray_netherlands
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 3 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 3 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 3 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 3 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 3 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##brc_microarray_usa
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 3 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 3 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 3 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 3 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 3 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_microarray_combined
python -u validation/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 3 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 3 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 3 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 3 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 3 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_rnaseq_syn2759792
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 3 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 3 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 3 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 3 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 3 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_rnaseq_syn4590909
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 3 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.01
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 3 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.05
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 3 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.1
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 3 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.2
python -u validation/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --clusters_output clusters.txt\
        --distance_threshold 3 --clusters_symbols clusters.symbols.txt --hidden_layers 3 --embedding_size 64 --p_val 0.3
#------------------------------------------------------------------------------------------------------------------------------------------------
deactivate                                          # Deactivate virtual environment