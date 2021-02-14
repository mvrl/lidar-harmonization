#!/bin/bash
 
#SBATCH -t 1-00:00:00 				    # Time for the job to run 
#SBATCH --job-name=LiDAR_JOIN     # Name of the job

#SBATCH --cpus-per-task=16
#SBATCH --partition=HAS24M128_S
#SBATCH --partition=HAS24M128_D
#SBATCH --partition=SAN16M64_S

#SBATCH --mail-type ALL
#SBATCH --mail-user nathan.jacobs@uky.edu
#SBATCH --account=col_nja224_uksr

module load ccs/anaconda/3

echo "Job $SLURM_JOB_ID running on SLURM NODELIST: $SLURM_NODELIST " 

. ../../setup_paths.sh

source activate laser 

echo "Joining validation data"

cat 01_24/val_full.csv |\
  sed -n '1!p' | python join.py | sort > val_joined.txt

echo "Joining test data"
cat 01_24/test_full.csv |\
  sed -n '1!p' | python join.py | sort > test_joined.txt

echo "Joining train data"
cat 01_24/train_full.csv |\
  sed -n '1!p' | python join.py | sort > train_joined.txt

echo "Making a single file"
cat *.txt | sort -u > all_joined.txt 

