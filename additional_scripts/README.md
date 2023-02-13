# Quick guide to the additional scripts

These are some scripts that came along during our work. 
This is not all that was used, as we simply utilized the python shell for most tasks regarding pandas and numpy, especially during the end. 
Still, we decided there is no real neccesity to remove any of these scripts, even if they were superceded or simply became unneeded.
Please note that there are some hardcoded paths in here that would need to be changed.
Other scripts may take arguments instead.

### Actively used (at least partially)
- `check_equal.py` checks Tox21 sets of Chemprop and Graphtrans for equality
- `ogb_split.py` reads the index lists of the OGB scaffold split and produces train, test and val datasets
- `tox21_split.py` generates data files from index files; includes `ogb_split.py`
- `open_pckl.py` reads a pickle file generated when training a model with the `--save_smiles_splits` parameter and produces train, test and val index files
- `plot_roc_auc.py` is used for plotting the micro- and macro-average ROC curves


### Legacy
- `tsv_to_csv.py` simply converts a tsv file to a csv
- `rm_invalid_smiles_preds.py` removes invalid rows from predictions
- `rounding.py` rounds predicted probabilities to 0 or 1 (automatically does `rm_invalid_smiles_preds.py` beforehand)
- `test_separation.py` creates a data set excluding the test set from OBG index files
