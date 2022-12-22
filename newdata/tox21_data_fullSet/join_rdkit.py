import os
import numpy as np
import pandas as pd
from functools import reduce
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem import PandasTools


### JOIN DATASETS
# get the ready-made SMILES of single toxicities and join them
os.chdir("./tox21_main")

nr_ahr = pd.read_csv("nr-ahr.smiles", sep="\t", names = ["SMILES", "ID", "NR-AhR"])
nr_ar = pd.read_csv("nr-ar.smiles", sep="\t", names = ["SMILES", "ID", "NR-AR"])
nr_ar_lbd = pd.read_csv("nr-ar-lbd.smiles", sep="\t", names = ["SMILES", "ID", "NR-AR-LBD"])
nr_aromatase = pd.read_csv("nr-aromatase.smiles", sep="\t", names = ["SMILES", "ID", "NR-Aromatase"])
nr_er = pd.read_csv("nr-er.smiles", sep="\t", names = ["SMILES", "ID", "NR-ER"])
nr_er_lbd = pd.read_csv("nr-er-lbd.smiles", sep="\t", names = ["SMILES", "ID", "NR-ER-LBD"])
nr_ppar_gamma = pd.read_csv("nr-ppar-gamma.smiles", sep="\t", names = ["SMILES", "ID", "NR-PPAR-gamma"])
sr_are = pd.read_csv("sr-are.smiles", sep="\t", names = ["SMILES", "ID", "SR-ARE"])
sr_atad5 = pd.read_csv("sr-atad5.smiles", sep="\t", names = ["SMILES", "ID", "SR-ATAD5"])
sr_hse = pd.read_csv("sr-hse.smiles", sep="\t", names = ["SMILES", "ID", "SR-HSE"])
sr_mmp = pd.read_csv("sr-mmp.smiles", sep="\t", names = ["SMILES", "ID", "SR-MMP"])
sr_p53 = pd.read_csv("sr-p53.smiles", sep="\t", names = ["SMILES", "ID", "SR-p53"])

dfs = [nr_ahr, nr_ar, nr_ar_lbd, nr_aromatase, nr_er, nr_er_lbd, nr_ppar_gamma, sr_are, sr_atad5, sr_hse, sr_mmp, sr_p53]
df_merged = reduce(lambda left,right: pd.merge(left,right,how='outer'), dfs)

# get the SMILES of the test set, converted from an SDF file
os.chdir("../tox21_testing/tox21_10k_challenge_test")
test = pd.read_csv("tox21_10k_challenge_test.smiles", sep="\t", names = ['SMILES', 'trash', 'ID', 'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'])
test = test.drop('trash', axis=1)

# merge to data from before
df_merged = pd.merge(df_merged, test, how = 'outer')

# lastly get data from the final eval set, split between a .smiles file and a .txt that hold the results
os.chdir("../../tox21_final_eval")

data = pd.read_csv("tox21_10k_challenge_score.smiles", sep="\t")
data.rename(columns = {'#SMILES':'SMILES', 'Sample ID':'ID'}, inplace = True)

res = pd.read_csv("tox21_10k_challenge_score.txt", sep="\t")
res.rename(columns = {'Sample ID':'ID'}, inplace = True)
res.replace('x', np.nan, inplace = True)

eval = pd.merge(data, res, how = 'outer')
eval[eval.columns.difference(['SMILES', 'ID'])] = eval[eval.columns.difference(['SMILES', 'ID'])].astype(float)

# merge to data from before
df_merged = pd.merge(df_merged, eval, how = 'outer')
df_merged = df_merged.drop('ID', axis=1)

os.chdir("../")



### JOIN STANDARDIZED SMILES INTO MERGED DATASET
df_merged.index += 1

standardized = pd.read_csv("tox21_merged_only_smiles-standardized.csv", sep='\t', names=["","newSMILES"])
standardized = standardized.set_index(standardized.columns[0])

df_merged = df_merged.join(standardized, how='inner')
#print(df_merged)
df_merged = df_merged.drop('SMILES', axis=1)
df_merged.rename(columns = {'newSMILES':'smiles'}, inplace = True)
first_column = df_merged.pop('smiles')
df_merged.insert(0, 'smiles', first_column)



### RDKIT WORKFLOW
# convert for RDkit, faulty SMILES will produce empty strings
PandasTools.AddMoleculeColumnToFrame(frame=df_merged,smilesCol='smiles',molCol='ROMol')

# drop the lines containing empty strings
df_merged['ROMol'].replace('', np.nan, inplace=True)
df_merged.dropna(subset=['ROMol'], inplace=True)

# define salts to be removed and apply
remover = SaltRemover(defnData="[Cl,Na,H20,K,Br,I]")
df_merged['ROMol'] = df_merged.apply(lambda x: remover.StripMol(x['ROMol']), axis=1)
# the above two lines may be substituted by the following, but salts can not be specified:
# PandasTools.RemoveSaltsFromFrame(frame=mols,molCol='ROMol')

# convert back to SMILES
# df_merged['ROMol'] = df_merged.apply(lambda x: Chem.MolToSmiles(x['ROMol']), axis=1)
# to get no explicit aromatic rings but kekulized mols, use:
df_merged['ROMol'] = df_merged.apply(lambda x: Chem.MolToSmiles(x['ROMol'], kekuleSmiles=True), axis=1)

# save
df_merged = df_merged.drop('smiles', axis=1)
df_merged = df_merged.rename(columns={'ROMol': 'smiles'})
first_column = df_merged.pop('smiles')
df_merged.insert(0, 'smiles', first_column)
df_merged.to_csv("tox21_complete.csv", index=False)
