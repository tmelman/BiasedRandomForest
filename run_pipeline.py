# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from randomforest import BiasedRandomForest
from ml_utils import (
    get_stratified_kfold_cval_splits,
    precision,
    recall,
    roc_curve,
    pr_curve,
    auc,
)

data_source_loc = '../data/diabetes_processed.csv'
figure_save_loc = '../results/results_fig.png'
# load PIMA data after pre-processing
pima = pd.read_csv(data_source_loc)
features = pima.columns.drop('Outcome')
data_x = pima[features]
data_y = pima['Outcome']

# create BRAF, 10fold cval
k = 10
folds = get_stratified_kfold_cval_splits(data_y, k)
precision_tr, recall_tr, roc_tr, pr_tr = [], [], [], []
precision_te, recall_te, roc_te, pr_te = [], [], [], []
hparams = {
    'max_depth': 100,
    'max_features': np.sqrt,
    'min_sample_leaf': 5,
    'k': 10,
    'p': .5,
    's': 100,
}
results = {}
for fold in np.unique(folds):
    test_inds = np.where(folds==[fold])
    train_inds = np.where(folds!=[fold])
    x_tr = data_x.iloc[train_inds]
    y_tr = data_y.iloc[train_inds]
    x_te = data_x.iloc[test_inds]
    y_te = data_y.iloc[test_inds]
    braf = BiasedRandomForest(data_x=x_tr, data_y=y_tr, hyper_params=hparams)
    braf.make_rfs()
    #predict 
    predictions_tr, predictions_te = [], []
    for i in range(len(x_tr)):
        dat = x_tr.iloc[i]
        predictions_tr.append(braf.predict(dat))
    for i in range(len(x_te)):
        dat = x_te.iloc[i]
        predictions_te.append(braf.predict(dat))
    labels_tr = y_tr.values.flatten()
    labels_te = y_te.values.flatten()
    predictions_tr = np.array(predictions_tr).flatten()
    predictions_te = np.array(predictions_te).flatten()
    precision_tr.append(precision(labels_tr,predictions_tr))
    precision_te.append(precision(labels_te,predictions_te))

    recall_tr.append(recall(labels_tr,predictions_tr))
    recall_te.append(recall(labels_te,predictions_te))

    roc_tr.append(roc_curve(labels_tr,predictions_tr))
    roc_te.append(roc_curve(labels_te,predictions_te))

    pr_tr.append(pr_curve(labels_tr,predictions_tr))
    pr_te.append(pr_curve(labels_te,predictions_te))


# print out precision, recall, ROC-AUC, PR-AUC
precision_tr_total = np.mean(np.array(precision_tr))
precision_te_total = np.mean(np.array(precision_te))

recall_tr_total = np.mean(np.array(recall_tr))
recall_te_total = np.mean(np.array(recall_te))

roc_tr_total = np.mean(np.array(roc_tr), axis=0)
roc_te_total = np.mean(np.array(roc_te), axis=0)

pr_tr_total = np.mean(np.array(pr_tr), axis=0)
pr_te_total = np.mean(np.array(pr_te), axis=0)

roc_auc_tr = auc(roc_tr_total)
roc_auc_te = auc(roc_te_total)

pr_auc_tr = auc(pr_tr_total)
pr_auc_te = auc(pr_te_total)

print(f'Train precision: {precision_tr_total}', end='\t')
print(f'Test precision: {precision_te_total}')
print(f'Train recall: {recall_tr_total}', end='\t')
print(f'Test recall: {recall_te_total}')
print(f'Train ROC AUC: {roc_auc_tr}', end='\t')
print(f'Test ROC AUC: {roc_auc_te}')
print(f'Train PR AUC: {pr_auc_tr}', end='\t')
print(f'Test PR AUC: {pr_auc_te}')

# show and save images for ROC, PR

matplotlib.rcParams['figure.figsize']=[15,10]
plt.subplot(2,2,1)
plt.plot(roc_tr_total[:,0], roc_tr_total[:,1])
plt.title(f'ROC on training set (AUC={roc_auc_tr:.2f})')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.subplot(2,2,2)
plt.plot(roc_te_total[:,0], roc_te_total[:,1])
plt.title(f'ROC on test set (AUC={roc_auc_te:.2f})')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.subplot(2,2,3)
plt.plot(pr_tr_total[:,0], pr_tr_total[:,1])
plt.title(f'PR on training set (AUC={pr_auc_tr:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.subplot(2,2,4)
plt.plot(pr_te_total[:,0], pr_te_total[:,1])
plt.title(f'PR on test set  (AUC={pr_auc_te:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig(figure_save_loc)
