#!/bin/bash

#SBATCH -o batch_params_wt.outfile
#SBATCH -e batch_params_wt.errfile
#SBATCH -t 98:00:00
#SBATCH -N 1
#SBATCH --mem 120gb

module load matlab/r2019a

matlab -nodesktop -nosplash -r "run_params_wt"
