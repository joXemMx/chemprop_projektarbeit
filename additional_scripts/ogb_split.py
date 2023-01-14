import pandas as pd
import os

# use pre-installed tox21 data set 
path = os.getcwd()
data = pd.read_csv(path+ '/data/tox21.csv')

# alternative to use the data set of GraphTrans
# path = '/beegfs/lo63tor/graphtrans/data/ogbg_moltox21/mapping/'
# data = pd.read_csv(path+'mol.csv.gz', compression='gzip')

# read split index files from OGB
splits_path = '/beegfs/lo63tor/graphtrans/data/ogbg_moltox21/split/scaffold/'
  
if not (os.path.isdir(path+ '/newdata/tox21_split_ogb/')):
  os.mkdir(path+ '/newdata/tox21_split_ogb/')

# save the data sets generated when applying the split index files
test_idx = pd.read_csv(splits_path+'test.csv.gz', compression='gzip').to_numpy().flatten()
test_data = pd.concat([data.iloc[[k]] for k in test_idx])
test_data.to_csv(path + 'newdata/tox21_split_ogb/ogb_test.csv', index=False)
del test_idx
del test_data

train_idx = pd.read_csv(splits_path+'train.csv.gz', compression='gzip').to_numpy().flatten()
train_data = pd.concat([data.iloc[[k]] for k in train_idx])
train_data.to_csv(path + 'newdata/tox21_split_ogb/ogb_train.csv', index=False)
del train_idx
del train_data

val_idx = pd.read_csv(splits_path+'valid.csv.gz', compression='gzip').to_numpy().flatten()
val_data = pd.concat([data.iloc[[k]] for k in val_idx])
val_data.to_csv(path + 'newdata/tox21_split_ogb/ogb_val.csv', index=False)
del val_idx
del val_data