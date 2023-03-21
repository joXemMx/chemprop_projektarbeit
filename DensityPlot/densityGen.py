import seaborn as sns
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

# GraphTrans Input
# split = ["chemSplit","OGBSplit","rndSplit"]
# path = ["chemsplit_chempropSplit","chemsplit-b50_chempropSplit","b50_OGBSplit","def_OGBSplit"]
# chosenSplit = split[1]
# chosenPath = path[3]

# df1 = pd.read_csv('../additional_scripts\GraphTransRes/'+chosenSplit+'/'+ chosenPath + '_result_pred.csv')
# df2 = pd.read_csv('../additional_scripts\GraphTransRes/'+chosenSplit+'/'+ chosenPath + '_result_test.csv')
# labels ziehen
labels = ['NR-AhR', 'NR-AR', 'NR-AR-LBD', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
# Chem Input
df1 = pd.read_csv('../tox21_checkpoints/full_set/opt_D-MPNN_rdkit_2d_normalized/test_preds.csv')
df2 = pd.read_csv('../newdata/tox21_split_full/test_full.csv')


# usage: python densityGen.py ../tox21_checkpoints\standard_set\D-MPNN_base/test_preds.csv ../newdata/tox21_split_chemprop/test_full.csv Name
# df1 = pd.read_csv(sys.argv[0])
# df2 = pd.read_csv(sys.argv[1])

zoom_on = False
# Wenn du mit sysArg machen willst
# name = sys.argv[2]
name = "density"


predSet_df = df1[labels].astype("float")
testSet_df = df2[labels].astype("float")
res = testSet_df.compare(predSet_df, keep_shape=True, keep_equal=True, result_names=("Real Value","Prediction"))
high_x = 0
low_x = 9999999999
high_y = 0
low_y = 9999999999
sum_all_x = 0
sum_all_y = 0
for label in labels:
    fig, ax = plt.subplots()
    zero_arr = np.zeros(len(res[label,"Real Value"]))
    ind = 0
    x = []
    y = []
    for value in res[label,"Real Value"]:
        if value == 1:
            x.append(res[label,"Prediction"][ind])
            if res[label,"Prediction"][ind] > high_x:
                high_x = res[label,"Prediction"][ind]
            if res[label,"Prediction"][ind] < low_x:
                low_x = res[label,"Prediction"][ind]
        elif value == 0:
            y.append(res[label,"Prediction"][ind])
            if res[label,"Prediction"][ind] < low_y:
                low_y = res[label,"Prediction"][ind]
            if res[label,"Prediction"][ind] > high_y:
                high_y = res[label,"Prediction"][ind]
        ind += 1
    print(label,":")
    sum_all_x += sum(x)/len(x)
    sum_all_y += sum(y)/len(y)
    print("AVG Val 1:", sum(x)/len(x))
    print("AVG Val 0:", sum(y)/len(y))
    print("High/Low x:", high_x,low_x)
    print("High/Low y:", high_y,low_y)
    print("hit:",len(x))
    print("no hit:",len(y))
    
    # # double hist
    # fig, (ax1, ax2) = plt.subplots(nrows=2)
    # def log_freq(weights):
    #     return np.log10(weights)
    # sns.histplot(x, bins=20, binrange=[0,1], alpha=0.5, color='red', ax=ax1, log_scale=(False,False))
    # sns.histplot(y, bins=20, binrange=[0,1], alpha=0.5, color='blue', ax=ax2, log_scale=(False,False))
    # fig.subplots_adjust(hspace=0.5)
    # ax1.set_ylabel('Count')
    # ax1.set_xlabel('Predictions for toxic')
    # ax2.set_ylabel('Frequency')
    # ax2.set_xlabel('Predictions for non-toxic')
    # plt.suptitle('Amount of predictions dependent on value')
    # plt.savefig('double_hist/hist_'+label+'.png')
    # plt.close()

    # # precision recall curve
    # arr = res[label,"Real Value"][~np.isnan(res[label,"Real Value"])]
    # percentile = sum(arr) / len(arr)
    # precision, recall, thresholds = precision_recall_curve(y_true = res[label,"Real Value"][~np.isnan(res[label,"Real Value"])], probas_pred = res[label,"Prediction"][~np.isnan(res[label,"Real Value"])])
    # plt.plot(recall, precision, label='PR Curve')
    # plt.axhline(y = percentile, color = 'r', linestyle = '--')
    # ax.legend(['PR curve', 'percentage of positive class'])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision - Recall Curve')
    # plt.savefig("precision_recall/pr_"+label+'.png')
    # plt.close()

    # # thresholds
    # thresholds = np.arange(0.1, 1.0001, 0.0001)
    # true_1_counts = []
    # true_0_counts = []
    # true_ones = []
    # false_ones = []
    # for threshold in thresholds:
    #     pred_true_thresh = np.round(x >= threshold)
    #     pred_false_thresh = np.round(y >= threshold)
    #     zeros_true = np.sum(pred_true_thresh == 0)
    #     ones_true = np.sum(pred_true_thresh == 1)
    #     zeros_false = np.sum(pred_false_thresh == 0)
    #     ones_false = np.sum(pred_false_thresh == 1)
    #     true_ones.append(ones_true)
    #     false_ones.append(ones_false)
    # plt.plot(thresholds, true_ones, label='correct toxic classification')
    # plt.plot(thresholds, false_ones, label='calse toxic classification')
    # plt.title('Amount of assigned toxic labels on different thresholds')
    # plt.xlabel('Threshold')
    # plt.ylabel('Count')
    # plt.xticks(np.arange(0.1,1.1,0.1))
    # plt.legend()
    # plt.savefig('threshold_plots/threshold_'+label+'.png')
    # plt.close()

    # # overlapping hist
    # sns.histplot(bins=20, binrange=[0,1], data=res[label][~np.isnan(res[label, "Real Value"])], palette="Set2", x="Prediction", hue="Real Value", log_scale=(False,True))
    # plt.title('Amount of predictions dependent on value in log scale')
    # plt.ylabel('log(Count)')
    # plt.savefig('overlap_hist/ohist_'+label+'.png')
    # plt.close()

    # for AUCs
    fpr, tpr, _ = roc_curve(res[label,"Real Value"][~np.isnan(res[label,"Real Value"])], res[label,"Prediction"][~np.isnan(res[label,"Real Value"])])
    plt.figure()
    plt.plot(
        fpr, tpr, 
        label="ROC curve (area = {0:0.2f})".format(roc_auc_score(res[label,"Real Value"][~np.isnan(res[label,"Real Value"])], res[label,"Prediction"][~np.isnan(res[label,"Real Value"])])),
        color="navy",
    )
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.title("ROC-AUC of " + label)
    plt.savefig('rocauc/roc_'+label+'.png')
    plt.close()