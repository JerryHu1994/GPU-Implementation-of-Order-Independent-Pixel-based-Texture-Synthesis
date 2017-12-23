#!/bin/sh
#SBATCH --partition=slurm_shortgpu
#SBATCH --time=0-00:05:00 # run time in days-hh:mm:ss
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8  # Use 8 OpenMP threads (you can use however many you like)
#SBATCH --error=/srv/home/jhu76/759--JerryHu1994/Project/src/ts_gpu.err
#SBATCH --output=/srv/home/jhu76/759--JerryHu1994/Project/src/ts_gpu.out
#SBATCH --gres=gpu:1
module load cuda/9
module load python/3.6.0
module load opencv/3.3.1
make
./ts_gpu texture1.jpg 400 3 11 2 3
