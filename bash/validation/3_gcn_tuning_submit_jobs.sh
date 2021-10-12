#!/bin/bash
cd $SCRATCH/gcncc.bioinformatics/slurm/brc_microarray_netherlands/validation/
sbatch 0_gcn_tuning.slurm

cd $SCRATCH/gcncc.bioinformatics/slurm/brc_microarray_usa/validation/
sbatch 0_gcn_tuning.slurm

cd $SCRATCH/gcncc.bioinformatics/slurm/scz_microarray_combined/validation/
sbatch 0_gcn_tuning.slurm

cd $SCRATCH/gcncc.bioinformatics/slurm/scz_rnaseq_syn2759792/validation/
sbatch 0_gcn_tuning.slurm

cd $SCRATCH/gcncc.bioinformatics/slurm/scz_rnaseq_syn4590909/validation/
sbatch 0_gcn_tuning.slurm

cd $SCRATCH/gcncc.bioinformatics/
