import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, roc_auc_score

act = pd.read_csv('/home/vo87poq/chemprop_projektarbeit/data/tox21_split/tox21_test.csv')
act = act[list(act)[1:]].fillna(0)
preds = pd.read_csv('/home/vo87poq/chemprop_projektarbeit/tox21_checkpoints/optimized_manualSplit/predictions.csv')
preds = preds[list(preds)[1:]]

score_roc_auc = roc_auc_score(act,preds)

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
    fpr[i], tpr[i], _ = roc_curve(act[list(act)[i]].to_numpy(), preds[list(preds)[i]].to_numpy())
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(act.to_numpy().ravel(), preds.to_numpy().ravel())
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
plt.title("Some extension of Receiver operating characteristic to multiclass")
plt.legend(loc="lower right")
plt.show()
plt.savefig('testplot.png')