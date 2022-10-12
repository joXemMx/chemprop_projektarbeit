import pandas as pd

data_path = '/beegfs/lo63tor/graphtrans/data/ogbg_moltox21/mapping'
splits_path = '/beegfs/lo63tor/graphtrans/data/ogbg_moltox21/split/scaffold/'

data = pd.read_csv(data_path+'mol.csv.gz', compression='gzip')

test_idx = pd.read_csv(splits_path+'test.csv.gz', compression='gzip').to_numpy().flatten()
test_data = pd.concat([data.iloc[[k]] for k in text_idx])
test_data.to_csv('tox21_test')
del test_idx
del test_data

train_idx = pd.read_csv(splits_path+'train.csv.gz', compression='gzip').to_numpy().flatten()
train_data = pd.concat([data.iloc[[k]] for k in train_idx])
train_data.to_csv('tox21_train')
del train_idx
del train_data

val_idx = pd.read_csv(splits_path+'valid.csv.gz', compression='gzip').to_numpy().flatten()
val_data = pd.concat([data.iloc[[k]] for k in val_idx])
val_data.to_csv('tox21_validation')
del val_idx
del val_data
