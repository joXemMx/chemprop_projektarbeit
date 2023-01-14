import re
import numpy as np
import pandas as pd

## Script used to join and save the SMILES with their classes
## Works only if your SMILES were given an ID (index+1) before standardization
## Additionally, determine number of duplicates

# txt = pd.read_csv('smiles.txt', names=['smiles'])
# arr = txt.duplicated().to_numpy()
# count = np.count_nonzero(arr)
# print("Duplicate SMILES: ",count)

smiles = pd.read_csv('smiles-standardized/smiles_with_id-standardized.csv', sep='\t', names=['ID', 'smiles'])

classes = pd.read_csv('tox21_compoundData.csv', usecols=["NR.AhR","NR.AR","NR.AR.LBD","NR.Aromatase","NR.ER","NR.ER.LBD","NR.PPAR.gamma","SR.ARE","SR.ATAD5","SR.HSE","SR.MMP","SR.p53"])
# rename NR.AhR to NR-AhR and so on
classes = classes.rename(columns=lambda x: re.sub('\.','-',x))
# create ID column
classes['ID'] = classes.index + 1
# merge on ID column
# drops rows with missing ID, i.e. where smiles standardization did not work
classes = pd.merge(classes, smiles, on='ID')
# same order as in smiles.txt, excluding SMILES where stand. did not work
classes.sort_values(by=['ID'])
# drop ID to get a clean dataset
classes.drop(['ID'], axis=1, inplace=True)

arr = classes.duplicated().to_numpy()
count = np.count_nonzero(arr)
print("Duplicate rows: ",count)

arr = classes['smiles'].duplicated().to_numpy()
count = np.count_nonzero(arr)
print("Duplicate SMILES: ",count)

classes.to_csv('tox21_full.csv', index=False)