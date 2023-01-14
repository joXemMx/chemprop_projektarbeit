import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


# Input
df1 = pd.read_csv('def_OGBSplit_result_pred.csv')
df2 = pd.read_csv('def_OGBSplit_result_test.csv')


testSet_df = df2
labels = testSet_df.columns.to_list()
predSet_df = df1[labels].astype("float")
res = testSet_df.compare(predSet_df, keep_shape=True, keep_equal=True, result_names=("Real Value","Prediction"))
for label in labels:
    zero_arr = np.zeros(len(res[label,"Real Value"]))
    ind = 0
    x = []
    y = []
    for value in res[label,"Real Value"]:
        if value == 1:
            x.append(res[label,"Prediction"][ind])   
        elif value == 0:
            y.append(res[label,"Prediction"][ind])   
        ind += 1
    print(label,":")
    print("hit:",len(x))
    print("no hit:",len(y))
    
    print("roc:",roc_auc_score(res[label,"Real Value"][~np.isnan(res[label,"Real Value"])], res[label,"Prediction"][~np.isnan(res[label,"Real Value"])]))
    print("roc with zero:",roc_auc_score(res[label,"Real Value"][~np.isnan(res[label,"Real Value"])],zero_arr[~np.isnan(res[label,"Real Value"])]))
    plot = sns.kdeplot(data=res[label], common_norm=False, palette="Set2", x="Prediction", hue="Real Value")
    leg = plt.legend(title=label,loc='upper right', labels=['toxic','not toxic'])
    leg.get_frame().set_linewidth(0.0)
    # plt.ylim(0, 14)
    plt.show()



    # df = pd.DataFrame([x,y]).transpose()
    # df = df.rename(columns={0: "hit", 1: "no hit"})
    # df = df.sort_values(by=['hit'], ascending=False)
    # # df.columns.values[0] = "hit"
    # # df.columns.values[1] = "no hit"
    # print(df.head())