#!/bin/bash
#SBATCH --partition=gpu_p100,gpu_v100,gpu_a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32000

source ~/miniconda3/etc/profile.d/conda.sh
conda activate chemprop
#python hyperparameter_optimization.py --data_path /home/vo87poq/chemprop_projektarbeit/data/tox21_split/tox21_train.csv --dataset_type classification --epochs=30 --seed=0 --num_iters=300 --split_type scaffold_balanced --config_save_path /home/vo87poq/chemprop_projektarbeit/data/tox21_split/config_300.txt --log_dir /home/vo87poq/chemprop_projektarbeit/data/tox21_split/logs_300
#python /home/vo87poq/chemprop_projektarbeit/hyperparameter_optimization.py --data_path /home/vo87poq/chemprop_projektarbeit/data/tox21.csv --dataset_type classification --epochs 30 --num_folds 10 --num_iters=20 --split_type scaffold_balanced --config_save_path /home/vo87poq/chemprop_projektarbeit/data/tox21_split/config_cv.txt --log_dir /home/vo87poq/chemprop_projektarbeit/data/tox21_split/logs_cv
python /home/vo87poq/chemprop_projektarbeit/train.py --data_path /home/vo87poq/chemprop_projektarbeit/data/tox21.csv --dataset_type classification --save_dir /home/vo87poq/chemprop_projektarbeit/tox21_checkpoints/standard_run --split_type scaffold_balanced --quiet