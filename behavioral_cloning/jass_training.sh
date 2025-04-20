#!/bin/bash -l
#SBATCH -p performance
#SBATCH -t 02:00:00
#SBATCH --mem=50g
#SBATCH --gpus=1
#SBATCH --job-name=JassTraining
singularity exec jass_training.sif python /home2/remo.aebi/Training_Scripts/IP5-Jass-AI/behavioral_cloning/jass_v1.py
