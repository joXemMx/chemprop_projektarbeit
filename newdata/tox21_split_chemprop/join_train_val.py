import pandas as pd

# create new set without the test set to allow hyperparameter optimization
# without the models seeing the test set
train = pd.read_csv('train_full.csv')
val = pd.read_csv('val_full.csv')

tv = pd.concat([train, val])
tv.to_csv('train+val_full.csv', index=False)