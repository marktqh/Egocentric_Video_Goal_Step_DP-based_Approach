#!/bin/bash -e

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=train_model

module purge

singularity exec --nv \
	    --overlay /scratch/qt2087/dsga1006/ego4d.ext3:ro \
	    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif\
	    /bin/bash -c "source /ext3/env.sh; python train_lavila_adapters_V5.py 512"