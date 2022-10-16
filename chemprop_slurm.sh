#!/bin/bash
#SBATCH --partition=gpu_p100,gpu_v100,gpu_a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=16000

source ~/miniconda3/etc/profile.d/conda.sh
conda activate chemprop
python /home/vo87poq/chemprop_projektarbeit/train.py --data_path /home/vo87poq/chemprop_projektarbeit/data/tox21_test_separation/tox21_train_val.csv --dataset_type classification --save_dir /home/vo87poq/chemprop_projektarbeit/tox21_checkpoints/noTest_10fold_gpu --split_type scaffold_balanced --num_folds 10 --quiet