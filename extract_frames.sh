#!/bin/bash -e

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=Extract_Frames
#SBATCH --array=0-348

module purge

singularity exec --nv \
	    --overlay /scratch/qt2087/dsga1006/ego4d.ext3:ro \
	    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif\
	    /bin/bash -c "source /ext3/env.sh; python process_frames_HPC.py ${SLURM_ARRAY_TASK_ID}"