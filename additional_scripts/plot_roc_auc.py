import pandas as pd
from sklearn import metrics

act = pd.read_csv('/home/vo87poq/chemprop_projektarbeit/data/tox21_split/tox21_test.csv')
preds = pd.read_csv('/home/vo87poq/chemprop_projektarbeit/tox21_checkpoints/optimized_manualSplit/predictions.csv')

score_roc_auc = metrics.roc_auc_score(act[list(act)[1:]].fillna(0),preds[list(preds)[1:]]))

## for every class:
fpr, tpr, thresholds = metrics.roc_curve(act[list(act)[2]].fillna(0).to_numpy(), preds[list(preds)[2]].to_numpy())
roc_auc = metrics.auc(fpr, tpr)
disp = metrics.RocCurveDisplay(fpr,tpr,roc_auc, 'some name')
display.plot()
plt.show()
