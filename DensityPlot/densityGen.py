import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# df = pd.read_csv('rndSplit_randomSplit_result.csv')
# df = pd.read_csv('graph-aug-moltox21-b50.csv')
df1 = pd.read_csv('def_OGBSplit_result_treshhold.csv')
df2 = pd.read_csv('../newdata/tox21_split_ogb/ogb_test.csv')
# df = pd.read_csv('test.csv')
# Create a density plot
# print(df)
print(len(df2))
print(len(df1))

arr_True = []
arr_False = []
for x in range(len(df2["NR-AhR"])):
    if df1["NR-AhR"][x] == df1["NR-AhR"][x]:
        if df1["NR-AhR"][x] == df2["NR-AhR"][x]:
            if  df1["NR-AhR"][x] == 0:
                arr_True.append(0)
            else:
                arr_True.append(1)
        else:
            if  df1["NR-AhR"][x] == 0:
                arr_False.append(0)
            else:
                arr_False.append(1)
print(len(arr_True))
print(len(arr_False))
# sns.distplot(df["NR-AhR"], rug=True, norm_hist=False,kde=True)
# sns.distplot(df["NR-AR-LBD"], rug=True, norm_hist=False,kde=True)
# sns.distplot(df["NR-ER"], rug=True, norm_hist=False,kde=True)
# sns.displot([arr_True,arr_False],kind="kde", legend=False,kernel='epa')
# sns.displot([arr_False],kind="kde")
sns.kdeplot([arr_True,arr_False], fill=False, linewidth=1)
plt.legend(loc='upper right', labels=['correct Hits', 'incorrect Hits'],kernel='epa')
# Create a density plot
#
# Show the plot
plt.show()