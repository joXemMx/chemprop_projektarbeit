import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
# df = pd.read_csv('rndSplit_randomSplit_result.csv')
df1 = pd.read_csv('graph-aug-moltox21-b50.csv')
# df1 = pd.read_csv('def_OGBSplit_result_treshhold.csv')
df2 = pd.read_csv('../newdata/tox21_split_ogb/ogb_test.csv')


# print(len(df2))
# print(len(df1))
# arr_True = []
# arr_False = []
# for x in range(len(df2["NR-AhR"])):
#     if df1["NR-AhR"][x] == df1["NR-AhR"][x]:
#         if df1["NR-AhR"][x] == df2["NR-AhR"][x]:
#             if  df1["NR-AhR"][x] == 0:
#                 arr_True.append(0)
#             else:
#                 arr_True.append(1)
#         else:
#             if  df1["NR-AhR"][x] == 0:
#                 arr_False.append(0)
#             else:
#                 arr_False.append(1)

testSet_df = df2.drop(columns="smiles")
labels = testSet_df.columns.to_list()
predSet_df = df1[labels].astype("float")
predSet_df.drop(predSet_df.tail(1).index,inplace=True)
test_df = pd.DataFrame([predSet_df["NR-AhR"],testSet_df["NR-AhR"]]).transpose()
test_df.columns.values[0] = "Prediction"
test_df.columns.values[1] = "Real"
# sns.kdeplot(data=test_df, bw_method=0.2,common_norm=False,x="Prediction", hue="Real", clip=(0,1), palette="Set2")

# print(testSet_df.head())
# print(predSet_df.head())
# print(len(testSet_df),len(predSet_df))
res = testSet_df.compare(predSet_df, keep_shape=True, keep_equal=True, result_names=("Real Value","Prediction"))
# res["NR-AhR"].plot.density()
pred_true_one = []
pred_true_zero = []
pred_false_one = []
pred_false_zero = []

for label in labels:
    ind = 0
    pred_true_one = 0
    pred_true_zero = 0
    pred_false_one = 0
    pred_false_zero = 0
    for value in res[label,"Prediction"]:
        if value == 0:
            if  res[label,"Real Value"][ind] == 0:
                # pred_true_zero.append(0)
                pred_true_zero += 1
            else:
                pred_false_zero+= 1
        else:
            if  res[label,"Real Value"][ind] == 1:
                pred_true_one+= 1
            else:
                pred_false_one+= 1
        ind += 1
    print(label,":")
    print("true1:",pred_true_one)
    print("true0:",pred_true_zero)
    print("false1:",pred_false_one)
    print("false0:",pred_false_zero)
    print("roc:",roc_auc_score(res[label,"Real Value"][~np.isnan(res[label,"Real Value"])], res[label,"Prediction"][~np.isnan(res[label,"Real Value"])]))
    plot = sns.kdeplot(data=res[label], bw_method=1,common_norm=False,x="Prediction", hue="Real Value", clip=(0,1), palette="Set2")
    plt.show()
# plot = sns.kdeplot(data=res["NR-AR"], bw_method=1,common_norm=False,x="Prediction", hue="Real Value", clip=(0,1), palette="Set2")



# for label in labels:
#    print(label)
#    plotx = sns.kdeplot(data=res[label], bw_method=1,common_norm=False,x="Prediction", hue="Real Value", clip=(0,1), palette="Set2")
#    plt.show()
# print(pred_true/len(res))
# print(pred_false/len(res))
# res_ploting = pd.DataFrame([pred_true,pred_false])
# print(res_ploting.head())
# sns.kdeplot(res[('NR-AR', 'Real Value')], label= "NR-AhR, correct")
# sns.kdeplot(res[('NR-AR', 'Prediction')], label= "NR-AhR, prediction")

# # plt.title('Density Plot with Multiple Labels')

# plt.xlabel('Toxicity')
# plt.ylabel('Density')
# plt.show()
#prop={'size': 16},
# hist_kws={'edgecolor':'black'},bins=int(180/5), color = 'darkblue', 