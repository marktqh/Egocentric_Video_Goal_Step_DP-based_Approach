#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=168:00:00
#SBATCH --mem=300GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=train_model_V9

module purge

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1200
export OMP_NUM_THREADS=8

singularity exec --nv \
    --overlay /scratch/qt2087/dsga1006/ego4d.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; torchrun --nproc_per_node=2 train_lavila_adapters_V9_multi.py 512"