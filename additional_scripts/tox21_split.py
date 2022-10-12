import pandas as pd

data_path = '../data/tox21.csv'
data = pd.read_csv(data_path)
#data_path = '/beegfs/lo63tor/graphtrans/data/ogbg_moltox21/mapping'
#data = pd.read_csv(data_path+'mol.csv.gz', compression='gzip')
splits_path = '/beegfs/lo63tor/graphtrans/data/ogbg_moltox21/split/scaffold/'

test_idx = pd.read_csv(splits_path+'test.csv.gz', compression='gzip').to_numpy().flatten()
test_data = pd.concat([data.iloc[[k]] for k in test_idx])
test_data.to_csv('../data/tox21_split/tox21_test.csv')
del test_idx
del test_data

train_idx = pd.read_csv(splits_path+'train.csv.gz', compression='gzip').to_numpy().flatten()
train_data = pd.concat([data.iloc[[k]] for k in train_idx])
train_data.to_csv('../data/tox21_split/tox21_train.csv')
del train_idx
del train_data

val_idx = pd.read_csv(splits_path+'valid.csv.gz', compression='gzip').to_numpy().flatten()
val_data = pd.concat([data.iloc[[k]] for k in val_idx])
val_data.to_csv('../data/tox21_split/tox21_validation.csv')
del val_idx
del val_data
