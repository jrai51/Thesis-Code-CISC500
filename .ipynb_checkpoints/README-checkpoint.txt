Title: Transformer-Based Continuous Authentication using Smartwatch IMU Data  
Author: Jagrit Rai  
Thesis Project - CISC 500, Queen's University  
April 2025

Overview
--------
This repository contains the experimental framework developed to evaluate the use of Transformer-based models for continuous user authentication using smartwatch IMU data.

The two frameworks evaluated are:
- MEMTO: Memory-Augmented Transformer for Unsupervised Anomaly Detection
- Anomaly Transformer: Self-attention-based model for time series anomaly detection

Both models are evaluated on their ability to distinguish genuine user typing behavior from impostor activity using smartwatch accelerometer and gyroscope data.

The key file for launching experiments is:
> `run_experiments.py`

Data & Setup
------------
Dataset:  
- The WACA dataset is used. It includes accelerometer and gyroscope data from 30 users typing predefined texts.
- The registration phase (Task-2) is used for training; the authentication phase (Task-1) is used for testing.
- Data should be placed in a folder named `WACA_dataset/` in CSV format as described in the thesis.

Environment Requirements:
- Python 3.8+
- Packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`
- Hardware: Experiments were run on a server with:
  - Intel Xeon Gold 6338 CPU
  - NVIDIA A40 GPU
  - 64 GB RAM

Installation (Optional):
- Create a virtual environment and install required packages via pip or conda.
- Ensure that the directories `MEMTO/`, `Anomaly-Transformer/`, and `WACA_dataset/` are present in the root directory.

Running Experiments
-------------------
The main script is `run_experiments.py`.

Basic command-line arguments:

python3 run_experiments.py --framework [MEMTO|Anomaly-Transformer] --experiment [train|inference|generate_results] --train_user USER_ID --test_user USER_ID

Examples:

1. Train and test MEMTO on user 1:
python3 run_experiments.py --framework MEMTO --experiment train --train_user 1 --test_user 1


2. Run inference on a trained Anomaly Transformer model:
python3 run_experiments.py --framework Anomaly-Transformer --experiment inference --train_user 1 --test_user 2


3. Run full evaluation (all user pairs) and generate result CSV:
python3 run_experiments.py --framework MEMTO --experiment generate_results


Modes:
- `train`: Trains and tests on specified user(s).
- `inference`: Performs inference on test user with trained model.
- `generate_results`: Runs all-user evaluation loop and saves results to CSV (`MEMTO_results.csv` or `AnomTrans_results.csv`).

Outputs
-------
- Train/test files are saved to `./MEMTO/data/WACA/` or `./Anomaly-Transformer/dataset/WACA/`
- Model outputs and console logs show training and test metrics.
- Final results are saved as:
  - `MEMTO_results.csv`
  - `AnomTrans_results.csv`
  
Each CSV contains:
gen_user,test_user,window_idx,anomaly_percentage

