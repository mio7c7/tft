#!/bin/bash
#SBATCH --job-name=my_python_job    # Job name
#SBATCH --output=output_atest.log          # Output file
#SBATCH --error=error_atest.log            # Error file
#SBATCH --partition=SCT   # Specify a partition (e.g., your_partition)
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=4         # Number of tasks
#SBATCH --gres=gpu:1
#SBATCH --mem=64G                    # Memory per node (e.g., 4 GB)

# Run the Python script
source venv/Scripts/activate
#python pytorch_forecasting/CPD/experiment.py --path 'R2_nonorm'
#python pytorch_forecasting/CPD/main_sr2.py --max_encoder_length 336 --quantile 0.975 --threshold_scale 1.25 --step 50 --outfile '975_125_25_default' --model_path '/default_r2_7d2d/trial_23/epoch=96.ckpt'
python pytorch_forecasting/CPD/ATtest.py --max_encoder_length 336 --method 'mae' --max_prediction_length 96 --quantile 0.975 --threshold_scale 1.25 --step 50 --outfile 'atest' --model_path '/default_o1_5d2d/trial_13/epoch=88.ckpt'