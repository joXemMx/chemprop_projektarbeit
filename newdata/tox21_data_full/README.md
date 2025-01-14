# Workflow of creating a bigger Tox21 dataset ###

tox21.sdf from http://bioinf.jku.at/research/DeepTox/tox21.html	(under 'Additional information on the data set')

tox21_compoundData.csv from http://bioinf.jku.at/research/DeepTox/tox21.html (under 'Additional information on the data set')

sdf translated to SMILES unsing: https://cactus.nci.nih.gov/translate/ --> smiles.txt

SMILES of smiles.txt standardized via PubChem produced the smiles-standardized folder.
One molecule could not be normalized and was lost (ID 7632, [SbH6+3].CC([O-])=O.CC([O-])=O.CC([O-])=O, FAILED_STANDARDIZATION).
To make use of our scripts, make sure to index the smiles.txt before standardization, simply going from 1 to 12707.


While in this directory, the further workflow using our scripts is as follows:

1) python join_smiles_data.py
2) python rdkit_filter.py tox21_full.csv tox21_full_desalted.csv
3) python group_and_join.py tox21_full_desalted.csv tox21_full_desalted_grouped
4) python check_pure_nans.py tox21_full_desalted_grouped_nans.csv

The resulting data set (tox21_full_desalted_grouped_nans.csv) has 8695 molecules.

Our result of "1)" was: Duplicate rows:  323, Duplicate SMILES:  3870

Our output of "2)":
```
Explicit valence for atom # 3 Si, 8, is greater than permitted
WARNING: not removing hydrogen atom without neighbors
Explicit valence for atom # 1 Si, 8, is greater than permitted
```
This means that we lost two more Molecules from our Dataset due to incopatibility with RDkit.
As Chemprop uses RDkit itself, these Molecules would be lost when put into Chemprop anyways.
After desalting, we get: Duplicate rows:  362, Duplicate SMILES:  4008

"3)" groups by SMILES, merges duplicate rows and solves conflics between different-value columns within the same group.
Note that 3) can take a little while, as it iterates over each group of SMILES.

"4)" checks if we have SMILES with only NaN values for the classes, in which case it holds no information and could be dropped.
Our result of 4) was 0. If this is 0, tox21_full_desalted_grouped_means.csv must have 0 per definition.
