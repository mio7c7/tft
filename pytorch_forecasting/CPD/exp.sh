#!/bin/bash
#SBATCH --job-name=ari    # Job name
#SBATCH --output=EncoderNormalizerrobust_r2_5d2d.log          # Output file
#SBATCH --error=EncoderNormalizerrobust_r2_5d2d.log            # Error file
#SBATCH --partition=SCT   # Specify a partition (e.g., your_partition)
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=4         # Number of tasks
#SBATCH --gres=gpu:1
#SBATCH --mem=32G                    # Memory per node (e.g., 4 GB)

# Run the Python script
source venv/Scripts/activate
python pytorch_forecasting/CPD/experiment.py --max_prediction_length 96
#python pytorch_forecasting/CPD/main_s.py --threshold_scale 1.2 --step 50 --outfile '99_12' --model_path '/no_normaliser/trial_11/epoch=49.ckpt'
