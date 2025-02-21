import subprocess
import argparse
from MEMTO.WACA_clean_data import *

def generate_train_test_files(train_user, test_user, framework="WACA"):
    # Call  functions from the other script to generate train.csv and test.csv

    # Generate Train file
    path_to_train = f"./WACA_dataset/user{train_user}_2.csv"
    save_user_train_data(path_to_train, path_to_csv="./MEMTO/data/WACA/WACA/train.csv")

    # Generate Test file
    path_to_test = f"./WACA_dataset/user{test_user}_1.csv"
    save_user_test_data(path_to_test, path_to_csv="./MEMTO/data/WACA/WACA/test.csv")
    save_user_test_label_data(impostor=False, path_to_csv="./MEMTO/data/WACA/WACA/test.csv", path_to_test_label="./MEMTO/data/WACA/WACA/test_label.csv")

    pass

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

    phase_type = None
    if mode == "memory_initial":
        phase_type = "second_train"
        num_epochs = 100

    elif mode == "test":
        phase_type = "test"

    cmd = ["python3", "main.py",
            "--mode", mode, 
            "--num_epochs", num_epochs,
            "--dataset", dataset, 
            "--data_path", f"./MEMTO/data/{dataset}/{dataset}/",
            "--input_c", input_c, 
            "--output_c", input_c, 
            "--n_memory", 10, 
            "--win_size", win_size, 
            "--memory_initial", memory_initial,
            "--phase_type", phase_type]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    # Optionally log result.stderr for errors
    return result.stdout

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", type=str, required=True)
    parser.add_argument("--train_user", type=str, required=True)
    parser.add_argument("--test_user", type=str, required=True)
    args = parser.parse_args()

    if args.framework == "MEMTO":
        # Generate train.csv and test.csv based on the specified users.
        generate_train_test_files(args.train_user, args.test_user, "MEMTO")

        # Run first training phase on train.csv
        train_output = run_MEMTO("train")

        # Run MEMTO second training phase
        second_train_output = run_MEMTO("memory_initial")

        # Run testing on test.csv
        test_output = run_MEMTO("test")

        # Process and record outputs as needed.
