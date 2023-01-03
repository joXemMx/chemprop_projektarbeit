import re
import numpy as np
import pandas as pd

## Script used to join and save the SMILES with their classes
## Additionally, determine number of duplicates

txt = pd.read_csv('smiles.txt', names=['smiles'])
arr = txt.duplicated().to_numpy()
count = np.count_nonzero(arr)
print("Duplicate SMILES: ",count)

classes = pd.read_csv('tox21_compoundData.csv', usecols=["NR.AhR","NR.AR","NR.AR.LBD","NR.Aromatase","NR.ER","NR.ER.LBD","NR.PPAR.gamma","SR.ARE","SR.ATAD5","SR.HSE","SR.MMP","SR.p53"])
classes = classes.rename(columns=lambda x: re.sub('\.','-',x))
classes.insert(0, 'smiles', txt)
arr = classes.duplicated().to_numpy()
count = np.count_nonzero(arr)
print("Duplicate rows: ",count)

classes.to_csv('tox21_full.csv', index=False)