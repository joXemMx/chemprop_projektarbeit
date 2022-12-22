import numpy as np
import pandas as pd

full = pd.read_csv("tox21_merged.csv")
full.index += 1

standardized = pd.read_csv("tox21_merged_only_smiles-standardized.csv", sep='\t', names=["","newSMILES"])
standardized = standardized.set_index(standardized.columns[0])

merged = full.join(standardized, how='inner')

merged = merged.drop('SMILES', axis=1)
merged.rename(columns = {'newSMILES':'smiles'}, inplace = True)
first_column = merged.pop('smiles')
merged.insert(0, 'smiles', first_column)

merged.to_csv("tox21_standardized.csv", index=False)