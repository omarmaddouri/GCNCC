The validation pipeline is designed for TAMU Terra cluster (Slurm batch system)

Note: The function calls that require internet connection (i.e. call mygene.MyGeneInfo())
should be executed on login nodes through bash scripts as the HPRC policy disables the internet connection on the compute nodes.

1) source 0_set_up_environment.sh to setup the python environment and install all the required dependencies
2) source 1_generate_network_files.sh to generate the common network files
3) source 2_gcn_tuning_split_files.sh to prepare the required data files before running the gcn embedding
4) Submit the embedding jobs by running source 3_gcn_tuning_submit_jobs.sh
5) Identify the best GCN architecture based on the validation loss for all the data sets (majority voting)
6) Tune the neighborhood threshold for geometric-AP by submitting jobs (source 4_clustering_tuning_submit_jobs.sh)
7) source 5_p_val_tuning_split_files.sh to generate datasets for different p-values for the best GCN architecture
8) Submit the embedding jobs by running source 6_p_val_tuning_embed_submit_jobs.sh
9) Submit the embedding jobs by running source 7_p_val_tuning_cluster_submit_jobs.sh
10) source 8_p_val_tuning_clusters_symbols.sh to generate the gene symbols for all the obtained clusters
11) source 9_p_val_tuning_scoring.sh to score the obtained clusters.
12) source 10_p_val_tuning_transformation.sh to transforme the original gene expression values to activity statistics.
13) source 11_p_val_tuning_selection_{classifier}.sh to select the discriminant clusters.