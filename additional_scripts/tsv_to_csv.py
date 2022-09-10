import pandas as pd
 
tsv_file='chemprop_projektarbeit/data/all-bio-structures-inchis-and-smiles.tsv'
 
# reading given tsv file
csv_table=pd.read_table(tsv_file,sep='\t')
 
# converting tsv file into csv
csv_table.to_csv('chemprop_projektarbeit/data/all-bio-structures-inchis-and-smiles.csv',index=False)
