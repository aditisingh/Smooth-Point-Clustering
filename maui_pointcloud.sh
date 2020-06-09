#!/bin/bash

#SBATCH -o pointcloud.outfile
#SBATCH -e pointcloud.errfile
#SBATCH -t 98:00:00
#SBATCH -N 1
#SBATCH --mem 120gb

module load matlab/r2019a

matlab -nodesktop -nosplash -r "benchmark"
