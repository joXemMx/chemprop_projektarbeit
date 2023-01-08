## check if tox21 datasets of GraphTrans and Chemprop are really equal
import pandas as pd

tpp_data = pd.read_csv('/beegfs/lo63tor/graphtrans/data/ogbg_moltox21/mapping/mol.csv.gz', compression='gzip').drop(columns=['mol_id'])
tpp_smiles = tpp_data.pop('smiles')
tpp_data.insert(0, 'smiles', tpp_smiles)
cp_data = pd.read_csv('../data/tox21.csv')

print('tox21 datasets of ToxPredProjekt and chemprop are equal') if tpp_data.equals(cp_data) else print('tox21 datasets of ToxPredProjekt and chemprop are NOT equal')
