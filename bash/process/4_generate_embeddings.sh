#!/bin/bash
cd $SCRATCH/gcncc.bioinformatics/slurm/brc_microarray_netherlands/process/
sbatch 0_embedding.slurm

cd $SCRATCH/gcncc.bioinformatics/slurm/brc_microarray_usa/process/
sbatch 0_embedding.slurm

cd $SCRATCH/gcncc.bioinformatics/slurm/scz_microarray_combined/process/
sbatch 0_embedding.slurm

cd $SCRATCH/gcncc.bioinformatics/slurm/scz_rnaseq_syn2759792/process/
sbatch 0_embedding.slurm

cd $SCRATCH/gcncc.bioinformatics/slurm/scz_rnaseq_syn4590909/process/
sbatch 0_embedding.slurm

cd $SCRATCH/gcncc.bioinformatics/
