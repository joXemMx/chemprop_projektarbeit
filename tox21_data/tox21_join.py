import os
import numpy as np
import pandas as pd
from functools import reduce

### get the ready-made SMILES of single toxicities and join them

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


### get the SMILES of the test set, converted from an SDF file

os.chdir("../tox21_testing/tox21_10k_challenge_test")

test = pd.read_csv("tox21_10k_challenge_test.smiles", sep="\t", names = ['SMILES', 'trash', 'ID', 'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'])
test.drop('trash', axis=1)

# merge to data from before
df_merged = pd.merge(df_merged, test, how = 'outer')


### lastly get data from the final eval set, split between a .smiles file and a .txt that hold the results

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

### save resulting dataframe

os.chdir("../")
df_merged.to_csv("tox21_merged.csv", index=False)