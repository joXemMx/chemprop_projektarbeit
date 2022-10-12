import pandas as pd

preds = pd.read_csv('../predictions/all_preds.csv')

## some rows produce "Invalid SMILES" instead of float values
# produce 0 if string is found, -1 else
rm_idx = preds['NR-AR'].astype(str).str.find('Invalid SMILES')
# get index of 0 values
rm_idx = rm_idx.index[rm_idx==0]
# drop 0 values
preds = preds.drop[preds.index[rm_idx]]
# finally, round up all numeric values left
preds[list(preds)[5:]] = preds[list(preds)[5:]].astype(float).round()

preds.to_csv('../predictions/all_preds_rounded.csv')
