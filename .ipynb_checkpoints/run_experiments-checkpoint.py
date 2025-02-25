import subprocess
import argparse
import time
from MEMTO.WACA_clean_data import *

def generate_train_test_files(train_user, test_user, framework="MEMTO", dataset="WACA"):
    # Call  functions from the other script to generate train.csv and test.csv
    data_path = ""
    if framework == "MEMTO":
        data_path = f"./MEMTO/data/{dataset}/{dataset}"
    elif framework == "Anomaly-Transformer":
        data_path = f"./Anomaly-Transformer/dataset/{dataset}"
        
        

    # Generate Train file
    path_to_train = f"./WACA_dataset/user{train_user}_2.csv"
    save_user_train_data(path_to_train, path_to_csv=f"{data_path}/train.csv")

    # Generate Test file
    path_to_test = f"./WACA_dataset/user{test_user}_1.csv"
    save_user_test_data(path_to_test, path_to_csv=f".{data_path}/test.csv")
    save_user_test_label_data(impostor=False, path_to_csv=f"{data_path}/test.csv", path_to_test_label=f"{data_path}/test_label.csv")

    pass

def run_AnomalyTransformer(mode, dataset="WACA", num_epochs=3, input_c=4, win_size=500):
    """
    
    train cmd: 
    python3 Anomaly-Transformer/main.py --anormly_ratio 1 --num_epochs 3    --batch_size 500  --mode train --dataset WACA  
            --data_path Anomaly-Transformer/dataset/WACA --input_c 4   --output_c 4
            
    test cmd:
    python3 main.py --anormly_ratio 1  --num_epochs 10  --batch_size 500     --mode test    --dataset WACA   --data_path dataset/WACA  --input_c 4    --output_c 4  

    """
    
    cmd = ["python3", "Anomaly-Transformer/main.py",
            "--mode", mode, 
            "--anormly_ratio", "1",
            "--num_epochs", str(num_epochs),
            "--dataset", dataset, 
            "--data_path", f"./Anomaly-Transformer/dataset/{dataset}",
            "--input_c", str(input_c), 
            "--output_c", str(input_c), 
            "--batch_size", str(win_size)]
    
    print(cmd)
    
    try:
        # Run the command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Print command output
        print("STDOUT:", result.stdout)
        return result.stdout
        
    except subprocess.CalledProcessError as e:
        # Print error details if the command fails
        print("ERROR: Command execution failed!")
        print("Return Code:", e.returncode)
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
    
    

def run_MEMTO(mode, dataset="WACA", num_epochs=10, input_c=4, win_size=1000):
    """
    Run MEMTO commands for training and testing.

    mode: "train", "memory_initial", or "test"
    dataset: name of dataset 
    input_c : number of channels in dataset
    win_size : number of rows included per window

    initial train:
    python3 MEMTO/main.py --anormly_ratio 1.0 --num_epochs 10   --batch_size 32  --mode train --dataset WACA  --data_path ./MEMTO/data/WACA/WACA/  --input_c 4 --output_c 4 --n_memory 10 --memory_initial False --win_size 1000 --phase_type None

    second train phase:
    python3 MEMTO/main.py --anormly_ratio 1.0 --num_epochs 100  --batch_size 32  --mode memory_initial --dataset WACA  --data_path ./MEMTO/data/WACA/WACA/  --input_c 4 --output_c 4 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --win_size 1000 --phase_type second_train

    test:
    python3 MEMTO/main.py --anormly_ratio 1.0 --num_epochs 10   --batch_size 32  --mode test --dataset WACA  --data_path ./MEMTO/data/WACA/WACA/  --input_c 4 --output_c 4 --n_memory 10 --memory_initial False --phase_type test

    """
    memory_initial = (mode == "memory_initial")

    phase_type = "None"
    if mode == "memory_initial":
        phase_type = "second_train"
        num_epochs = 100

    elif mode == "test":
        phase_type = "test"

    cmd = ["python3", "MEMTO/main.py",
            "--mode", mode, 
            "--num_epochs", str(num_epochs),
            "--dataset", dataset, 
            "--data_path", f"./MEMTO/data/{dataset}/{dataset}/",
            "--input_c", str(input_c), 
            "--output_c", str(input_c), 
            "--n_memory", "10", 
            "--win_size", str(win_size), 
            "--memory_initial", str(memory_initial),
            "--phase_type", phase_type]
    
    print(cmd)
    
    try:
        # Run the command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Print command output
        print("STDOUT:", result.stdout)
        return result.stdout
        
    except subprocess.CalledProcessError as e:
        # Print error details if the command fails
        print("ERROR: Command execution failed!")
        print("Return Code:", e.returncode)
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
    # Optionally log result.stderr for errors
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", type=str, required=True)
    parser.add_argument("--train_user", type=str, required=True)
    parser.add_argument("--test_user", type=str, required=True)
    parser.add_argument("--no_retrain", action="store_true", help="Skip retraining if a trained model exists.")
    args = parser.parse_args()

    if args.framework == "MEMTO":
        # Generate train.csv and test.csv based on the specified users.
        generate_train_test_files(args.train_user, args.test_user, "MEMTO")
        if not args.no_retrain:
            print("Training MEMTO model...")
            train_output = run_MEMTO("train")

            print("Running MEMTO second training phase...")
            second_train_output = run_MEMTO("memory_initial")
        else:
            print("Skipping MEMTO training: model already exists.")
       

        # Run testing on test.csv
        test_output = run_MEMTO("test")

        # Process and record outputs as needed.
    elif args.framework == "Anomaly-Transformer":
        # Generate train.csv and test.csv based on the specified users.
        generate_train_test_files(args.train_user, args.test_user, framework="Anomaly-Transformer")
        
        if not args.no_retrain:
            print("Training Anomaly Transformer model...")
            train_output = run_AnomalyTransformer("train")
        else:
            print("Skipping Anomaly Transformer training: model already exists.")

        print("Running Anomaly Transformer test...")
        test_output = run_AnomalyTransformer("test")
