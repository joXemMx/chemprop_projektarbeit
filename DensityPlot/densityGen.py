import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

# Input
split = ["chemSplit","OGBSplit","rndSplit"]
path = ["chemsplit_chempropSplit","chemsplit-b50_chempropSplit","b50_OGBSplit","def_OGBSplit"]
chosenSplit = split[1]
chosenPath = path[3]
df1 = pd.read_csv('../additional_scripts\GraphTransRes/'+chosenSplit+'/'+ chosenPath + '_result_pred.csv')
df2 = pd.read_csv('../additional_scripts\GraphTransRes/'+chosenSplit+'/'+ chosenPath + '_result_test.csv')


testSet_df = df2
labels = testSet_df.columns.to_list()
predSet_df = df1[labels].astype("float")
res = testSet_df.compare(predSet_df, keep_shape=True, keep_equal=True, result_names=("Real Value","Prediction"))
for label in labels:
    fig, ax = plt.subplots()
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
    sns.kdeplot(data=res[label], common_norm=False, palette="Set2", x="Prediction", hue="Real Value", ax=ax)
    leg = plt.legend(title=label,loc='upper right', labels=['toxic','not toxic'])
    leg.get_frame().set_linewidth(0.0)


    axins = zoomed_inset_axes(ax, 4, loc='center right', borderpad=5)
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")
    axins.set_xlim([0.04,0.055])
    axins.set_ylim([0,25])
    
    sns.kdeplot(data=res[label], common_norm=False, palette="Set2", x="Prediction", hue="Real Value", ax=axins)
    axins.set(ylabel=None, xlabel=None, yticklabels=[])
    axins.xaxis.tick_top()
    axins.legend().remove()
    # ax2 = plt.axes([0.6, 0.2, .5, .5])
    # sns.kdeplot(data=res[label], common_norm=False, palette="Set2", x="Prediction", hue="Real Value", ax=ax2,axes=False)
    # ax2.set_title('zoom')
    # ax2.set_xlim([0.05,0.2])
    # ax2.set_ylim([0,14])
    # axins = zoomed_inset_axes(plot, zoom=4, loc=5)
    # axins.plot(ax=ax[1])
    # axins.set_xlim(0,1)
    # axins.set_ylim(0, 14)
    # axins.plot(data)
    plt.savefig(chosenPath + "_" +  label)
    plt.show()

    # df = pd.DataFrame([x,y]).transpose()
    # df = df.rename(columns={0: "hit", 1: "no hit"})
    # df = df.sort_values(by=['hit'], ascending=False)
    # # df.columns.values[0] = "hit"
    # # df.columns.values[1] = "no hit"
    # print(df.head())