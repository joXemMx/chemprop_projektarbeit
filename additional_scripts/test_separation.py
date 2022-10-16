import pandas as pd
import numpy as np
import os

data_path = '../data/tox21.csv'
data = pd.read_csv(data_path)
#data_path = '/beegfs/lo63tor/graphtrans/data/ogbg_moltox21/mapping'
#data = pd.read_csv(data_path+'mol.csv.gz', compression='gzip')
splits_path = '/beegfs/lo63tor/graphtrans/data/ogbg_moltox21/split/scaffold/'

os.mkdir('../data/tox21_test_separation/')

test_idx = pd.read_csv(splits_path+'test.csv.gz', compression='gzip').to_numpy().flatten()
test_data = pd.concat([data.iloc[[k]] for k in test_idx])
test_data.to_csv('../data/tox21_split/tox21_test.csv', index=False)
del test_idx
del test_data

train_idx = pd.read_csv(splits_path+'train.csv.gz', compression='gzip').to_numpy().flatten()
val_idx = pd.read_csv(splits_path+'valid.csv.gz', compression='gzip').to_numpy().flatten()
train_val_idx = np.concatenate([train_idx, val_idx])
train_val = pd.concat([data.iloc[[k]] for k in train_val_idx])
train_val.to_csv('../data/tox21_test_separation/tox21_train_val.csv', index=False)
