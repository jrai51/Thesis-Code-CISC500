Usage:

For MEMTO
- initial train on WACA dataset
python3 MEMTO/main.py --anormly_ratio 1.0 --num_epochs 10   --batch_size 32  --mode train --dataset WACA  --data_path ./MEMTO/data/WACA/WACA/  --input_c 4 --output_c 4 --n_memory 10 --memory_initial False --win_size 1000 --phase_type None

- second train phase:

python3 MEMTO/main.py --anormly_ratio 1.0 --num_epochs 100  --batch_size 32  --mode memory_initial --dataset WACA  --data_path ./MEMTO/data/WACA/WACA/  --input_c 4 --output_c 4 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --win_size 1000 --phase_type second_train

- test phase: 

python3 MEMTO/main.py --anormly_ratio 1.0 --num_epochs 10   --batch_size 32  --mode test --dataset WACA  --data_path ./MEMTO/data/WACA/WACA/  --input_c 4 --output_c 4 --n_memory 10 --memory_initial False --phase_type test