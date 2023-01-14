import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem import PandasTools

# read SMILES data
mols = pd.read_csv(str(sys.argv[1]))

# convert for RDkit, faulty SMILES will produce empty strings
PandasTools.AddMoleculeColumnToFrame(frame=mols,smilesCol='smiles',molCol='ROMol')

# drop the lines containing empty strings
mols['ROMol'].replace('', np.nan, inplace=True)
mols.dropna(subset=['ROMol'], inplace=True)

# define salts to be removed and apply
remover = SaltRemover(defnData="[Cl,Na,H20,K,Br,I]")
mols['ROMol'] = mols.apply(lambda x: remover.StripMol(x['ROMol']), axis=1)
# the above two lines may be substituted by the following, but salts can not be specified:
# PandasTools.RemoveSaltsFromFrame(frame=mols,molCol='ROMol')

# convert back to SMILES
#mols['ROMol'] = mols.apply(lambda x: Chem.MolToSmiles(x['ROMol'], isomericSmiles=True), axis=1)
# to get no explicit aromatic rings but kekulized mols, use:
mols['ROMol'] = mols.apply(lambda x: Chem.MolToSmiles(x['ROMol'], kekuleSmiles=True), axis=1)

# save
mols = mols.drop('smiles', axis=1)
mols = mols.rename(columns={'ROMol': 'smiles'})
first_column = mols.pop('smiles')
mols.insert(0, 'smiles', first_column)
mols.to_csv(str(sys.argv[2]), index=False)
