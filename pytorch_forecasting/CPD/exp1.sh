#!/bin/bash
#SBATCH --job-name=my_python_job    # Job name
#SBATCH --output=output_r2.log          # Output file
#SBATCH --error=error_r2.log            # Error file
#SBATCH --partition=SCT   # Specify a partition (e.g., your_partition)
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=4         # Number of tasks
#SBATCH --gres=gpu:1
#SBATCH --mem=32G                    # Memory per node (e.g., 4 GB)

# Run the Python script
source venv/Scripts/activate
#python pytorch_forecasting/CPD/experiment.py --path 'R2_nonorm'
#python pytorch_forecasting/CPD/main_sr2.py --max_encoder_length 240 --quantile 0.975 --threshold_scale 1.25 --step 50 --outfile '975_125_15_5d_r2' --model_path '/no_normaliser_r2_5d/trial_6/epoch=47.ckpt'
python pytorch_forecasting/CPD/ATtest.py --max_encoder_length 240 --max_prediction_length 96 --quantile 0.975 --threshold_scale 1.25 --step 50 --outfile '975_125_15_ENR5d2d_r2' --model_path '/EncoderNormalizerrobust_r2_5d2d/trial_17/epoch=49.ckpt'