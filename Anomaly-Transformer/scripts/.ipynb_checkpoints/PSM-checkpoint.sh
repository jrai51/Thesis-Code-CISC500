export CUDA_VISIBLE_DEVICES=0

python3 main.py --anormly_ratio 1 --num_epochs 3    --batch_size 256  --mode train --dataset PSM  --data_path dataset/PSM --input_c 25    --output_c 25
python3 main.py --anormly_ratio 1  --num_epochs 10       --batch_size 256     --mode test    --dataset PSM   --data_path dataset/PSM  --input_c 25    --output_c 25  --pretrained_model 20


python3 main.py --anormly_ratio 1 --num_epochs 3    --batch_size 500  --mode train --dataset WACA  --data_path dataset/WACA --input_c 4   --output_c 4
python3 main.py --anormly_ratio 1  --num_epochs 10  --batch_size 500     --mode test    --dataset WACA   --data_path dataset/WACA  --input_c 4    --output_c 4  

