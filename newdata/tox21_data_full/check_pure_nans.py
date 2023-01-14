import sys
import pandas as pd
import numpy as np

## Check if any SMILES row has only np.nan values in the columns
## In this case, it would hold no information and could be dropped

df = pd.read_csv(str(sys.argv[1]))

# Create a boolean mask indicating which values are null
mask = df.isnull()

# Use the mask to select only the columns you want to consider (ignoring the "smiles" column)
mask = mask.loc[:, mask.columns != "smiles"]

# Check which rows are all null
all_null = mask.all(axis=1)

# Use the mask to select the rows of the original dataframe that are all null
filtered_df = df[all_null]

# Count the number of rows in the filtered dataframe
num_rows = len(filtered_df)

print("Number of smiles without any class data: ", num_rows) 