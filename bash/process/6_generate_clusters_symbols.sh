#!/bin/bash
module load Python/3.6.6-intel-2018b
cd $SCRATCH/gcncc.bioinformatics                    # Go to project directory
source venv/bin/activate                            # Activate virtual environment (Command line should show environment name on left)

## Generate clusters symbols

##brc_microarray_netherlands

##Clustering method: ig
python -u process/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method ig --embedding_method ge\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
        
##Clustering method: geometric_ap
python -u process/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method geometric_ap --embedding_method gcn\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method geometric_ap --embedding_method kernel_pca\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method geometric_ap --embedding_method vae\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt 
python -u process/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method geometric_ap --embedding_method ge\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt        
##Clustering method: standard_ap
python -u process/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method standard_ap --embedding_method gcn\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method standard_ap --embedding_method kernel_pca\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method standard_ap --embedding_method vae\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt 
##Clustering method: mcl
python -u process/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method mcl --embedding_method gcn\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method mcl --embedding_method kernel_pca\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method mcl --embedding_method vae\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt         
##Clustering method: wgcna
python -u process/3_generate_clusters_symbols.py --dataset brc_microarray_netherlands --clustering_method wgcna --embedding_method ge\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt          
#------------------------------------------------------------------------------------------------------------------------------------------------
##brc_microarray_usa

##Clustering method: ig
python -u process/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method ig --embedding_method ge\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
        
##Clustering method: geometric_ap
python -u process/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method geometric_ap --embedding_method gcn\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method geometric_ap --embedding_method kernel_pca\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method geometric_ap --embedding_method vae\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt 
python -u process/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method geometric_ap --embedding_method ge\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt        
##Clustering method: standard_ap
python -u process/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method standard_ap --embedding_method gcn\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method standard_ap --embedding_method kernel_pca\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method standard_ap --embedding_method vae\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt 
##Clustering method: mcl
python -u process/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method mcl --embedding_method gcn\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method mcl --embedding_method kernel_pca\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method mcl --embedding_method vae\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt         
##Clustering method: wgcna
python -u process/3_generate_clusters_symbols.py --dataset brc_microarray_usa --clustering_method wgcna --embedding_method ge\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt        
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_microarray_combined

##Clustering method: ig
python -u process/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method ig --embedding_method ge\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
        
##Clustering method: geometric_ap
python -u process/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method geometric_ap --embedding_method gcn\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method geometric_ap --embedding_method kernel_pca\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method geometric_ap --embedding_method vae\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method geometric_ap --embedding_method ge\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt        
##Clustering method: standard_ap
python -u process/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method standard_ap --embedding_method gcn\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method standard_ap --embedding_method kernel_pca\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method standard_ap --embedding_method vae\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt 
##Clustering method: mcl
python -u process/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method mcl --embedding_method gcn\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method mcl --embedding_method kernel_pca\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method mcl --embedding_method vae\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt      
##Clustering method: wgcna
python -u process/3_generate_clusters_symbols.py --dataset scz_microarray_combined --clustering_method wgcna --embedding_method ge\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt        
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_rnaseq_syn2759792

##Clustering method: ig
python -u process/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method ig --embedding_method ge\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
        
##Clustering method: geometric_ap
python -u process/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --embedding_method gcn\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --embedding_method kernel_pca\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --embedding_method vae\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method geometric_ap --embedding_method ge\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt        
##Clustering method: standard_ap
python -u process/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method standard_ap --embedding_method gcn\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method standard_ap --embedding_method kernel_pca\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method standard_ap --embedding_method vae\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt 
##Clustering method: mcl
python -u process/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method mcl --embedding_method gcn\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method mcl --embedding_method kernel_pca\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method mcl --embedding_method vae\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt    
##Clustering method: wgcna
python -u process/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn2759792 --clustering_method wgcna --embedding_method ge\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt        
#------------------------------------------------------------------------------------------------------------------------------------------------
##scz_rnaseq_syn4590909

##Clustering method: ig
python -u process/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method ig --embedding_method ge\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
        
##Clustering method: geometric_ap
python -u process/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --embedding_method gcn\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --embedding_method kernel_pca\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --embedding_method vae\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method geometric_ap --embedding_method ge\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt        
##Clustering method: standard_ap
python -u process/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method standard_ap --embedding_method gcn\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method standard_ap --embedding_method kernel_pca\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method standard_ap --embedding_method vae\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt 
##Clustering method: mcl
python -u process/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method mcl --embedding_method gcn\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method mcl --embedding_method kernel_pca\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt
python -u process/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method mcl --embedding_method vae\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt        
##Clustering method: wgcna
python -u process/3_generate_clusters_symbols.py --dataset scz_rnaseq_syn4590909 --clustering_method wgcna --embedding_method ge\
        --clusters_output clusters.txt --clusters_symbols clusters.symbols.txt        
#------------------------------------------------------------------------------------------------------------------------------------------------
deactivate                                          # Deactivate virtual environment