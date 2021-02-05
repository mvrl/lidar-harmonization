#!/bin/bash

# V.Gazula 2/15/2019
 
#SBATCH -t 10:05:00   				#Time for the job to run 
#SBATCH --job-name=Matlab_Singularity   	#Name of the job

#SBATCH -n 1					#Number of cores needed for the job
#SBATCH --partition=KCS				#Name of the queue

#SBATCH --mail-type ALL				#Send email on start/end
#SBATCH --mail-user gazula@uky.edu		#Where to send email
#SBATCH --account=gol_griff_uksr		#Name of account to run under

#SBATCH --output=Output.out 			#Name of output file

#Module needed for this Matlab job
module purge
module load  ccs/matlab/R2018b

echo "Job $SLURM_JOB_ID running on SLURM NODELIST: $SLURM_NODELIST " 
#Singularity Matlab Program execution command 
matlab -nodisplay -nodesktop -r 'prime; exit ' 
