#!/bin/bash -l
#SBATCH -p performance
#SBATCH -t 20:00:00
#SBATCH --mem=50g
#SBATCH --gpus=2
#SBATCH --cpus-per-task=2
#SBATCH --job-name=JassTraining
singularity exec jass_training.sif python /home2/remo.aebi/Training_Scripts/IP5-Jass-AI/behavioral_cloning/v2/jass_v2.py
