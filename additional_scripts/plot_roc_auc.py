import sys
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer

## example call for Chemprop:
## python plot_roc_auc.py path/to/test/set path/to/model/directory/that/includes/predictions MyTitle
## the plot will automatically be saved in path/to/model/directory/that/includes/predictions

## example call for Chemprop:
## python plot_roc_auc.py path/to/test/set path/to/predictions MyTitle path/to/save/plot/in

## to get all the plots (with the same name) from your model directories to a sinlge folder, run this nice and simple line:
## find tox21_checkpoints/ -name '*.png' -exec sh -c 'cp "$1" "auc_plots/$(basename $(dirname $(dirname "$1")) )_$(basename $(dirname "$1"))_auc_plot.png"' _ {} \; 
## this will copy all auc plots from the checkpoints directory to the auc_plots folder and name them in this fashion:
## The plot in ``tox21_checkpoints/full_set/D-MPNN_base/auc_plot.png`` will be named ``full_set_D-MPNN_base_auc_plot.png``

#### legacy comments
# GraphTrans Process
# model = ('./GraphTransRes/')
# splits = ['chemSplit/','OGBSplit/', 'rndSplit/']
# models = ['chemsplit_chempropSplit','chemsplit-b50_chempropSplit','def_OGBSplit','b50_OGBSplit','dev_gcn','rndSplit_80_10_10_randomSplit','rndSplit-b50_80_10_10_randomSplit']
# act = pd.read_csv(model+splits[0] + models[0] + '_result_test.csv')
# act = act[list(act)[0:]]
# preds = pd.read_csv(model+splits[0]+ models[0] + '_result_pred.csv')
# preds = preds[list(preds)[0:]]

###########################

# act should be path including the csv
act = pd.read_csv(sys.argv[1])
# if og_act_len = 12
og_act_len = len(act.columns)
if (og_act_len != 12):
    act = act[list(act)[1:]]


if (og_act_len != 12):
    # model has to be specified as the directory the model is in
    # i.e. "tox21_checkpoints/standard_set/D-MPNN_base"
    model = str(sys.argv[2])
    preds = pd.read_csv(model + '/fold_0/test_preds.csv')
    preds = preds[list(preds)[1:]]
else:
    # pred should be path including the csv
    pred = str(sys.argv[2])
    preds = pd.read_csv(pred)

#score_roc_auc = roc_auc_score(act,preds)

## for one class:
# fpr, tpr, thresholds = roc_curve(act[list(act)[1]].to_numpy(), preds[list(preds)[1]].to_numpy())
# roc_auc = auc(fpr, tpr)
# disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='some name')
# disp.plot()
# plt.show()
# plt.title("Receiver operating characteristic")
# plt.savefig('testplot.png')

# for every class
# following: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
n_classes = len(act.columns)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    a = act[list(act)[i]].to_numpy()
    p = preds[list(preds)[i]].to_numpy()
    fpr[i], tpr[i], _ = roc_curve(a[~np.isnan(a)], p[~np.isnan(a)])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
a = act.to_numpy().ravel()
p = preds.to_numpy().ravel()
fpr["micro"], tpr["micro"], _ = roc_curve(a[~np.isnan(a)], p[~np.isnan(a)])
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure()
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=3,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=3,
)

# NUM_COLORS = 12
# cm = plt.get_cmap('gist_rainbow')
# colors=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
# for i, color in zip(range(n_classes), colors):
#     plt.plot(
#         fpr[i],
#         tpr[i],
#         color=color,
#         lw=1,
#         label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
#     )

plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.title(str(sys.argv[3]))

if (og_act_len != 12):
    plt.savefig(model+'/auc_plot.png')
else: 
    plt.savefig(sys.argv[4])
plt.show()
