#!/bin/bash
#SBATCH --job-name=cnn_lstm_study       # Job name
#SBATCH --partition=gpu_h100            # Partition name (change to gpu_l40s)
#SBATCH --mem=128G
#SBATCH --output=$HOME/MatchMentorData/cnn_lstm_optuna_study_runs/job_%j.out
#SBATCH --error=$HOME/MatchMentorData/cnn_lstm_optuna_study_runs/job_%j.err

# Activate env
source $HOME/mambaforge/etc/profile.d/conda.sh
conda activate matchmentor

# Run the Python module
python -u $HOME/MatchMentor/cnn_lstm/cnn_lstm.py
