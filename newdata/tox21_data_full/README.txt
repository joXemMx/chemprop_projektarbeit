tox21.sdf from 			http://bioinf.jku.at/research/DeepTox/tox21.html	(under 'Additional information on the data set')
tox21_compoundData.csv from 	http://bioinf.jku.at/research/DeepTox/tox21.html	(under 'Additional information on the data set')
translated to SMILES unsing: 	https://cactus.nci.nih.gov/translate/ 			--> smiles.txt

while in this directory, the workflow is as follows:
1) python join_txt_csv.py
2) python group_and_join.py tox21_full.csv tox21_grouped