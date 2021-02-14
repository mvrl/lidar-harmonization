#!/bin/bash
 
#SBATCH -t 1-00:00:00 				    # Time for the job to run 
#SBATCH --job-name=LiDAR_INDEX   	# Name of the job

#SBATCH --cpus-per-task=16
#SBATCH --partition=SAN16M64_M    # Name of the queue

#SBATCH --mail-type ALL
#SBATCH --mail-user nathan.jacobs@uky.edu
#SBATCH --account=col_nja224_uksr

#SBATCH --output=output.out 			# Name of output file

module load ccs/anaconda/3

echo "Job $SLURM_JOB_ID running on SLURM NODELIST: $SLURM_NODELIST " 

. ../../setup_paths.sh
python process.py

