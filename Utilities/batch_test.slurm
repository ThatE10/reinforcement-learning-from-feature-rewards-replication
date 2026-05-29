#!/bin/bash
#SBATCH --job-name=batch_size_test
#SBATCH --output=/users/enguye17/reinforcement-learning-from-feature-rewards-replication/Data/logs/batch_test_%j.out
#SBATCH --error=/users/enguye17/reinforcement-learning-from-feature-rewards-replication/Data/logs/batch_test_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:4
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --partition=Nebula_GPU

module load cuda/12.8

/users/enguye17/reinforcement-learning-from-feature-rewards-replication/.venv/bin/python3 \
    /users/enguye17/reinforcement-learning-from-feature-rewards-replication/Utilities/generate_model_response.py