#!/bin/bash

#SBATCH -t 1-00:00:00 				    # Time for the job to run 
#SBATCH --job-name=ENTWINE_BUILD_2   	# Name of the job

#SBATCH --cpus-per-task=8
#SBATCH --partition=HAS24M128_S # Name of the queue

#SBATCH --mail-type ALL
#SBATCH --mail-user nathan.jacobs@uky.edu
#SBATCH --account=col_nja224_uksr
#SBATCH --array=1-64 # NOTE: this currently must be 1 indexed

#SBATCH --output=log-j2_subset-%A_%a.out

echo "Job $SLURM_JOB_ID running on SLURM NODELIST: $SLURM_NODELIST " 

source ./config.sh

module load ccs/anaconda/3

conda activate entwine

entwine build -i ${SCAN_NAME}/scan.json -o ${ENTWINE_OUT} -t 8 \
  -s ${SLURM_ARRAY_TASK_ID} ${SLURM_ARRAY_TASK_MAX}

