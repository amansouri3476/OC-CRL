#!/bin/bash

#SBATCH --partition=$SLURM_PARTITION
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=6G
#SBATCH --time=0:30:00
#SBATCH --output=./slurm_out/sparse_clevr-%j.out
#SBATCH --error=./slurm_err/sparse_clevr-%j.err

module load singularity

# python3 /home/user/kubric/sparse_clevr_generator.py --max_workers 10 --num_samples 1000 --start_id 5000
python3 /home/user/kubric/sparse_clevr_generator.py --worker_script $1 --max_workers $2 --num_samples $3 --start_id $4 --global_start_id $5 --force_regen $6 --properties_list $7 --fixed_properties_list $8 --subdirectory $9 --number_of_objects ${10} --CLEVR_OBJECTS ${11}

module purge
