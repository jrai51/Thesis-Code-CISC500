import subprocess
import argparse
import time
import os
from WACA_clean_data import *
import re

def generate_train_test_files(train_user, test_user, framework="MEMTO", dataset="WACA"):
    # Call  functions from the other script to generate train.csv and test.csv
    data_path = ""
    if framework == "MEMTO":
        data_path = f"./MEMTO/data/{dataset}/{dataset}"
    elif framework == "Anomaly-Transformer":
        data_path = f"./Anomaly-Transformer/dataset/{dataset}"

    # Get filepaths
    path_to_train = f"./WACA_dataset/user{train_user}_2.csv"
    path_to_test = f"./WACA_dataset/user{test_user}_1.csv"
    
    data_exists = True
    # check if user file exists
    if not os.path.exists(path_to_train):
        print(f"{path_to_train} does not exist.")
        data_exists = False
    elif not os.path.exists(path_to_test):
        print(f"{path_to_test} does not exist.")
        data_exists = False
        
    if not data_exists: return data_exists
    
    # Generate train file 
    save_user_train_data(path_to_train, path_to_csv=f"{data_path}/train.csv")

    # Generate Test file
    save_user_test_data(path_to_test, path_to_csv=f"{data_path}/test.csv")
    # if the train user is the same as test user, impostor=False
    save_user_test_label_data(impostor=(not train_user == test_user), path_to_csv=f"{data_path}/test.csv", path_to_test_label=f"{data_path}/test_label.csv")

    pass

def run_AnomalyTransformer(mode, dataset="WACA", num_epochs=3, input_c=6, win_size=1000, anormly_ratio=2, batch_size=8):
    """
    
    train cmd: 
    python3 Anomaly-Transformer/main.py --anormly_ratio 1 --num_epochs 3    --batch_size 500  --mode train --dataset WACA  
            --data_path Anomaly-Transformer/dataset/WACA --input_c 4   --output_c 4
            
    test cmd:
    python3 main.py --anormly_ratio 1  --num_epochs 10  --batch_size 500     --mode test    --dataset WACA   --data_path dataset/WACA  --input_c 4    --output_c 4  

    """
    
    cmd = ["python3", "Anomaly-Transformer/main.py",
            "--mode", mode, 
            "--anormly_ratio", str(anormly_ratio),
            "--num_epochs", str(num_epochs),
            "--dataset", dataset, 
            "--data_path", f"./Anomaly-Transformer/dataset/{dataset}",
            "--input_c", str(input_c), 
            "--output_c", str(input_c), 
            "--win_size", str(win_size),
            "--batch_size", str(batch_size)
          ]
    
    print("Running command:", cmd)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("STDOUT:", result.stdout)
        
        if mode == "inference_experiment":
            # Parse the output to extract anomaly percentages.
            anomaly_percentages = []
            # Look for lines with "anomaly_percentage="
            for line in result.stdout.splitlines():
                # For example, line: "Window 3: accuracy=0.0220, anomaly_percentage=0.02, imposter decision=False"
                m = re.search(r"anomaly_percentage=([\d\.]+)", line)
                if m:
                    anomaly_percentages.append(float(m.group(1)))
            return anomaly_percentages
        
        elif mode == "inference_with_sums":
            anomaly_percentages = []
            # Look for lines with "anomaly_percentage="
            for line in result.stdout.splitlines():
                # For example, line: "Window 3: accuracy=0.0220, anomaly_percentage=0.02, imposter decision=False"
                m = re.search(r"energy_sum=([\d\.]+)", line)
                if m:
                    anomaly_percentages.append(float(m.group(1)))
            return anomaly_percentages
        else:
            print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print("ERROR: Command execution failed!")
        print("Return Code:", e.returncode)
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return None
    
    
    

def run_MEMTO(mode, dataset="WACA", num_epochs=35, input_c=6, win_size=1000, anormly_ratio=10, n_memory_items=10):
    """
    Run MEMTO commands for training and testing.

    mode: "train", "memory_initial", or "test"
    dataset: name of dataset 
    input_c : number of channels in dataset
    win_size : number of rows included per window

    initial train:
    python3 MEMTO/main.py --anormly_ratio 1.0 --num_epochs 10   --batch_size 32  --mode train --dataset WACA  --data_path ./MEMTO/data/WACA/WACA/  --input_c 4 --output_c 4 --n_memory 10 --memory_initial False --win_size 1000 --phase_type None
    
    OR IS IT:
    python3 MEMTO/main.py --mode train --anormly_ratio 1 --num_epochs 100 --dataset WACA --data_path ./MEMTO/data/WACA/WACA/ --input_c 6 --output_c 6 --n_memory 10 --batch_size 32 --memory_initial False --phase_type None

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
            "--anormly_ratio", str(anormly_ratio),
            "--num_epochs", str(num_epochs),
            "--dataset", dataset, 
            "--data_path", f"./MEMTO/data/{dataset}/{dataset}/",
            "--input_c", str(input_c), 
            "--output_c", str(input_c), 
            "--n_memory", str(n_memory_items), 
            "--win_size", str(win_size), 
            "--memory_initial", str(memory_initial),
            "--phase_type", phase_type]
    
    print("Running command:", cmd)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("STDOUT:", result.stdout)
        
        if mode == "inference_experiment":
            # Parse the output to extract anomaly percentages.
            anomaly_percentages = []
            # Look for lines with "anomaly_percentage="
            for line in result.stdout.splitlines():
                # For example, line: "Window 3: accuracy=0.0220, anomaly_percentage=0.02, imposter decision=False"
                m = re.search(r"anomaly_percentage=([\d\.]+)", line)
                if m:
                    anomaly_percentages.append(float(m.group(1)))
            return anomaly_percentages
        
        elif mode == "inference_with_sums":
            # Parse the output to extract anomaly percentages.
            anomaly_percentages = []
            # Look for lines with "anomaly_percentage="
            for line in result.stdout.splitlines():
                # For example, line: "Window 3: accuracy=0.0220, anomaly_percentage=0.02, imposter decision=False"
                m = re.search(r"energy_sum=([\d\.]+)", line)
                if m:
                    anomaly_percentages.append(float(m.group(1)))
            return anomaly_percentages
        else:
            print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print("ERROR: Command execution failed!")
        print("Return Code:", e.returncode)
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return None
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="train")
    parser.add_argument("--framework", type=str, required=True)
    parser.add_argument("--train_user", type=str, default=1)
    parser.add_argument("--test_user", type=str, default=1)
    parser.add_argument("--anormly_ratio", type=int, default=1)
    parser.add_argument("--channel_size", type=int, default=6)
    parser.add_argument("--no_retrain", action="store_true", help="Skip retraining if a trained model exists.")
    
    args = parser.parse_args()

    # Conditionally require train_user and test_user unless experiment == "generate_results"
    if args.experiment != "generate_results":
        if args.train_user is None or args.test_user is None:
            parser.error("--train_user and --test_user are required for experiments other than generate_results.")

    user_ids = [1,2,3,4,5,6,7,8, 19, 21, 22, 26,27,28,29] + [x for x in range(35, 50) if x != 47 ]    
    print("user count:", len(user_ids))
    
    # this outer if/else case will have to be refactored, probably not best practice
    if args.experiment == "train":
        if args.framework == "MEMTO":
            # Generate train.csv and test.csv based on the specified users.
            generate_train_test_files(args.train_user, args.test_user, "MEMTO")
            if not args.no_retrain:
                print("Training MEMTO model...")
                train_output = run_MEMTO("train", anormly_ratio=args.anormly_ratio)

                print("Running MEMTO second training phase...")
                second_train_output = run_MEMTO("memory_initial", anormly_ratio=args.anormly_ratio)
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
                train_output = run_AnomalyTransformer("train", input_c=args.channel_size, anormly_ratio=args.anormly_ratio)
            else:
                print("Skipping Anomaly Transformer training: model already exists.")

            print("Running Anomaly Transformer test...")
            test_output = run_AnomalyTransformer("test", input_c=args.channel_size)
    
    
    
    elif args.experiment == "inference":
        # INCOMPLETE: NEED TO SPECIFY WHICH USER WAS TRAINED ON, OR ENSURE THE PROVIDED TRAIN_USER WAS USED IN TRAINING
        # test user is fine
        if args.framework == "MEMTO":
            generate_train_test_files(args.train_user, args.test_user, "MEMTO") # generate train/test files 
            train_output = run_MEMTO("train", anormly_ratio=args.anormly_ratio) # train model on appropriate user
            run_MEMTO(mode="inference_with_sums") # Run inferencing 
            
        elif args.framework == "Anomaly-Transformer":
            generate_train_test_files(args.train_user, args.test_user, "Anomaly-Transformer") # generate train/test files 
            train_output = run_AnomalyTransformer("train",  input_c=args.channel_size, anormly_ratio=args.anormly_ratio) # train model on appropriate user
            run_AnomalyTransformer(mode="inference_with_sums") # Run inferencing with SUMS
            
            
            
    elif args.experiment == "generate_results" and args.framework == "MEMTO": #will have to come up with a better way to switch the model methods. Use class methods?

        results_file = open("MEMTO_results.csv", "w")  # Open the results file in write mode.
        # print a header to the file
        results_file.write("gen_user,test_user,window_idx,anomaly_percentage\n")  # Writing headers for the columns
        results_file.flush()  # Ensure header is immediately written to the file
        
        for gen_user in user_ids:     
            # Generate train.csv and test.csv based on the specified users.
            generate_train_test_files(gen_user, gen_user, "MEMTO")
            
            # train a model on the genuine user
            print("Training MEMTO model...")
            train_output = run_MEMTO("train", anormly_ratio=args.anormly_ratio)

            print("Running MEMTO second training phase...")
            second_train_output = run_MEMTO("memory_initial", anormly_ratio=args.anormly_ratio)
            
            # Could save a model for each user at this point
            
            for test_user in user_ids:
                # inference against the impostor user windows, no need to re-train
                print(f"gen_user: {gen_user}, test user: {test_user}")
                generate_train_test_files(gen_user, test_user, "MEMTO")
                
                window_results = run_MEMTO(mode="inference_with_sums") # I need this to return a list of anomaly percentages
                print(window_results)
        
                # For each window, record its anomaly percentage.
                # For each window's anomaly percentage, record a row
                for idx, anomaly_pct in enumerate(window_results):
                    # Create results row 
                    row = f"{gen_user},{test_user},{idx},{anomaly_pct}\n" 
                    results_file.write(row)  # Write the row to the results file
                    results_file.flush()  # Immediately flush data to disk after each write to ensure it's saved
              
        # After processing all users, close the results file
        results_file.close()
        print("Saved aggregated scores for all users to MEMTO_results.csv")
    
    elif args.experiment == "generate_results" and args.framework == "Anomaly-Transformer": #will have to come up with a better way to switch the model methods. Use class methods?

        results_file = open("AnomTrans_results.csv", "w")  # Open the results file in write mode.
        # print a header to the file
        results_file.write("gen_user,test_user,window_idx,anomaly_percentage\n")  # Writing headers for the columns
        results_file.flush()  # Ensure header is immediately written to the file
        
        for gen_user in user_ids:     
            # Generate train.csv and test.csv based on the specified users.
            generate_train_test_files(gen_user, gen_user, "Anomaly-Transformer")
            
            # train a model on the genuine user
            print("Training Anomaly Transformer model...")
            train_output = run_AnomalyTransformer("train") # train model on appropriate user
            
            # Could save a model for each user at this point?
            
            for test_user in user_ids:
                # inference against the impostor user windows, no need to re-train
                print(f"gen_user: {gen_user}, test user: {test_user}")
                generate_train_test_files(gen_user, test_user, "Anomaly-Transformer")
                
                window_results = run_AnomalyTransformer(mode="inference_with_sums") # Run inferencing 
                print(window_results)
        
                # For each window, record its anomaly percentage.
                # For each window's anomaly percentage, record a row
                for idx, anomaly_pct in enumerate(window_results):
                    # Create results row 
                    row = f"{gen_user},{test_user},{idx},{anomaly_pct}\n" 
                    results_file.write(row)  # Write the row to the results file
                    results_file.flush()  # Immediately flush data to disk after each write to ensure it's saved
              
        # After processing all users, close the results file
        results_file.close()
        print("Saved aggregated scores for all users to AnomTrans_results.csv")
    
    
        
                

               
        
