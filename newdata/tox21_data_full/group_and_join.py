import sys
import pandas as pd
import numpy as np

## Script used to group and join duplicate rows and SMILES

# read in the CSV file and group it by the "smiles" column
df = pd.read_csv(str(sys.argv[1]))
groups = df.groupby('smiles')

# create a new empty dataframe to store the aggregated data
agg_df = pd.DataFrame()
agg_nan_df = pd.DataFrame()

# iterate over the groups and create a new row for each group
for name, group in groups:
  row = {}
  nan_row = {}
  
  # iterate over the remaining columns
  for col in df.columns:
    if col == 'smiles':
      # for the "smiles" column, we just want to save the value from the group
      row[col] = name
      nan_row[col] = name
    else:
      # get all the non-nan values for this column in the group
      values = group[col].dropna().values

      # if there are no non-nan values, save nan for this column
      if len(values) == 0:
        row[col] = np.nan
        nan_row[col] = np.nan
      # if all the values are the same, save that value for this column
      elif all(values == values[0]):
        row[col] = values[0]
        nan_row[col] = values[0]
      # otherwise, save the mean value for this column (regression)
      # or take nan as a compromise (classification)
      else:
        row[col] = np.mean(values)
        nan_row[col] = np.nan
  
  # add the row to the aggregated dataframe
  agg_df = agg_df.append(row, ignore_index=True)
  agg_nan_df = agg_nan_df.append(nan_row, ignore_index=True)

# save the aggregated dataframe to a CSV file
first_column = agg_df.pop('smiles')
agg_df.insert(0, 'smiles', first_column)
first_column = agg_nan_df.pop('smiles')
agg_nan_df.insert(0, 'smiles', first_column)

agg_df.to_csv(str(sys.argv[2])+'_means.csv', index=False)
agg_nan_df.to_csv(str(sys.argv[2])+'_nans.csv', index=False)