import pandas as pd
import os

data_path = os.getcwd()+'/data/'
print(data_path)
data = pd.read_csv(data_path+ 'tox21.csv')
#data_path = '/beegfs/lo63tor/graphtrans/data/ogbg_moltox21/mapping'
#data = pd.read_csv(data_path+'mol.csv.gz', compression='gzip')

splits_path = '/beegfs/lo63tor/graphtrans/data/ogbg_moltox21/split/scaffold/'
#splits_path = '../data/tox21_split_chemprop/'

if (splits_path == '../data/tox21_split_chemprop/'):
  
  test_idx = pd.read_csv(splits_path+'tox21_test_idx.csv').to_numpy().flatten()
  test_data = pd.concat([data.iloc[[k]] for k in test_idx])
  test_data.to_csv('../data/tox21_split_chemprop/tox21_test.csv', index=False)
  del test_idx
  del test_data

  train_idx = pd.read_csv(splits_path+'tox21_train_idx.csv').to_numpy().flatten()
  train_data = pd.concat([data.iloc[[k]] for k in train_idx])
  train_data.to_csv('../data/tox21_split_chemprop/tox21_train.csv', index=False)
  del train_idx
  del train_data

  val_idx = pd.read_csv(splits_path+'tox21_valid_idx.csv').to_numpy().flatten()
  val_data = pd.concat([data.iloc[[k]] for k in val_idx])
  val_data.to_csv('../data/tox21_split_chemprop/tox21_validation.csv', index=False)
  del val_idx
  del val_data
  
else:
  
  if not (os.path.isdir(data_path+ 'tox21_split/')):
    os.mkdir(data_path+'tox21_split/')

  test_idx = pd.read_csv(splits_path+'test.csv.gz', compression='gzip').to_numpy().flatten()
  test_data = pd.concat([data.iloc[[k]] for k in test_idx])
  test_data.to_csv( data_path + 'tox21_split/tox21_test.csv', index=False)
  del test_idx
  del test_data

  train_idx = pd.read_csv(splits_path+'train.csv.gz', compression='gzip').to_numpy().flatten()
  train_data = pd.concat([data.iloc[[k]] for k in train_idx])
  train_data.to_csv( data_path + 'tox21_split/tox21_train.csv', index=False)
  del train_idx
  del train_data

  val_idx = pd.read_csv(splits_path+'valid.csv.gz', compression='gzip').to_numpy().flatten()
  val_data = pd.concat([data.iloc[[k]] for k in val_idx])
  val_data.to_csv(data_path + 'tox21_split/tox21_validation.csv', index=False)
  del val_idx
  del val_data
