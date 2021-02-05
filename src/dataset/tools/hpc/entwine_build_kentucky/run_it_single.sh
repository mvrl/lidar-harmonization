#!/bin/bash
 
#SBATCH -t 14-00:00:00 				    # Time for the job to run 
#SBATCH --job-name=ENTWINE_BUILD   	# Name of the job

#SBATCH --cpus-per-task=32
#SBATCH --partition=SKY32M192_L # Name of the queue

#SBATCH --mail-type ALL
#SBATCH --mail-user nathan.jacobs@uky.edu
#SBATCH --account=col_nja224_uksr

#echo "Job $SLURM_JOB_ID running on SLURM NODELIST: $SLURM_NODELIST " 

LAZ_IN=/pscratch/nja224_uksr/ky_lidar/laz_raw/N055E33*.laz
ENTWINE_OUT=/pscratch/nja224_uksr/ky_lidar/entwine_all
SCAN_NAME=ky_data

module load ccs/anaconda/3

conda activate entwine

rm -rf ${ENTWINE_OUT}
rm -rf ${SCAN_NAME}

mkdir -p ${ENTWINE_OUT} 

entwine scan -i ${LAZ_IN} -o ${SCAN_NAME} -r EPSG:3089 EPSG:3089 --hammer -t 8

entwine build -i ${SCAN_NAME}/scan.json -o ${ENTWINE_OUT} -t 4 -s 1 4 &
entwine build -i ${SCAN_NAME}/scan.json -o ${ENTWINE_OUT} -t 4 -s 2 4 &
entwine build -i ${SCAN_NAME}/scan.json -o ${ENTWINE_OUT} -t 4 -s 3 4 &
entwine build -i ${SCAN_NAME}/scan.json -o ${ENTWINE_OUT} -t 4 -s 4 4 &

echo "Waiting for build subset jobs to finish..."

wait

entwine merge ${ENTWINE_OUT}
