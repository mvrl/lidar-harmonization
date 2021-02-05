#!/bin/bash
 
#SBATCH -t 1-00:00:00 				    # Time for the job to run 
#SBATCH --job-name=FIX_HEADER   	# Name of the job

#SBATCH --cpus-per-task=8
#SBATCH --partition=HAS24M128_S   # Name of the queue

#SBATCH --mail-type ALL
#SBATCH --mail-user nathan.jacobs@uky.edu
#SBATCH --account=col_nja224_uksr

#SBATCH --output=log-%A_%a.out

#SBATCH --array=1-32

module load ccs/anaconda/3

echo "Job $SLURM_JOB_ID running on SLURM NODELIST: $SLURM_NODELIST " 

. ../../setup_paths.sh

# TODO should make this work whether or not *ARRAY_TASK* is set

# 1) Get all LAZ files
# 2) Stable sort
# 3) Split into chunks
# 4) Process junks using multiprocessing

find ${LAZPATH} -name '*.laz' -printf "%f\n" | \
  sort -s | \
  split -n r/${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_MAX} | \
  python process.py

