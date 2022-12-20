#!/bin/bash
#SBATCH --partition=gpu_p100,gpu_v100,gpu_a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32000

source ~/miniconda3/etc/profile.d/conda.sh
conda activate chemprop
#python hyperparameter_optimization.py --data_path /home/vo87poq/chemprop_projektarbeit/data/tox21_split/tox21_train.csv --dataset_type classification --epochs=30 --seed=0 --num_iters=300 --split_type scaffold_balanced --config_save_path /home/vo87poq/chemprop_projektarbeit/data/tox21_split/config_300.txt --log_dir /home/vo87poq/chemprop_projektarbeit/data/tox21_split/logs_300
#python /home/vo87poq/chemprop_projektarbeit/hyperparameter_optimization.py --gpu 0 --empty_cache --num_iters 100 --data_path /home/vo87poq/chemprop_projektarbeit/data/tox21_split_chemprop/tox21_train+val.csv --dataset_type classification --split_type scaffold_balanced --config_save_path /home/vo87poq/chemprop_projektarbeit/tox21_checkpoints/optimization/standard_set/config.json
#python /home/vo87poq/chemprop_projektarbeit/train.py --data_path /home/vo87poq/chemprop_projektarbeit/tox21_data/tox21_merged.csv --dataset_type classification --split_type scaffold_balanced --save_dir /home/vo87poq/chemprop_projektarbeit/tox21_checkpoints/full_set/D-MPNN --quiet
python /home/vo87poq/chemprop_projektarbeit/train.py --gpu 0 --data_path /home/vo87poq/chemprop_projektarbeit/data/tox21.csv --dataset_type classification --split_type random --save_dir /home/vo87poq/chemprop_projektarbeit/tox21_checkpoints/standard_set/D-MPNN_randomSplit --quiet --save_smiles_splits


#python predict.py --test_path data/tox21_split/tox21_test.csv --checkpoint_dir tox21_checkpoints/tox21_split --preds_path tox21_checkpoints/tox21_split/predictions.csv
#python predict.py --test_path data/tox21_split/tox21_test.csv --checkpoint_dir tox21_checkpoints/standard_run --preds_path tox21_checkpoints/standard_run/predictions.csv
#python /home/to85vet/Projects/chemprop_projektarbeit/additional_scripts/plot_roc_auc.py