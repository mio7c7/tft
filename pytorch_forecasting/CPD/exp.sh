#!/bin/bash
#SBATCH --job-name=my_python_job    # Job name
#SBATCH --output=main_s_output.log          # Output file
#SBATCH --error=main_serror.log            # Error file
#SBATCH --partition=SCT   # Specify a partition (e.g., your_partition)
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=4         # Number of tasks
#SBATCH --gres=gpu:1
#SBATCH --mem=16G                    # Memory per node (e.g., 4 GB)

# Run the Python script
source venv/Scripts/activate
python pytorch_forecasting/CPD/main_s.py
