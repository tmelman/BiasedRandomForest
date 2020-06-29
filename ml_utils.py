import numpy as np
import scipy as scp

def get_stratified_train_test_split(y, ntrain):
    # Assumes y is binary
    y = y.flatten()
    all_indices = range(len(y))
    neg_class_indices = np.where(y==0)[0]
    pos_class_indices = np.setxor1d(all_indices, neg_class_indices)
    n_neg_train = int(np.round(ntrain * len(neg_class_indices)/len(y)))
    n_pos_train = ntrain - n_neg_train
    train_neg = np.random.choice(neg_class_indices, n_neg_train,replace=False)
    train_pos = np.random.choice(pos_class_indices, n_pos_train,replace=False)
    train_indices = np.concatenate((train_neg, train_pos))
    test_indices = np.setxor1d(all_indices, train_indices)
    return train_indices, test_indices

def get_stratified_kfold_cval_splits(y, k=10):
    y = y.flatten()
    n_each_fold = int(len(y)/k)
    # indices_left = np.array(range(len(y)))
    fold_membership = np.zeros(len(y))-1
    for i in range(k-1):
        y_unassigned = np.array(range(len(y)))[fold_membership==-1]
        inds_ith_fold, _ = get_stratified_train_test_split(y[y_unassigned], n_each_fold)
        fold_membership[y_unassigned[inds_ith_fold]] = i
    fold_membership[fold_membership==-1] = k-1
    return fold_membership

def precision(labels, predictions):
    # true positives/predicted positives
    true_positives = sum(np.logical_and(labels, predictions))
    predicted_positives = sum(predictions)
    return true_positives/predicted_positives

def recall(labels, predictions):
    # true positives/labeled positives
    true_positives = sum(np.logical_and(labels, predictions))
    positives = sum(labels)
    return true_positives/positives

def sensitivity(labels, predictions):
    return recall(labels, predictions)

def specificity(labels, predictions):
    false_positives = sum(np.logical_and(labels, 1-predictions))
    negatives = sum(1-predictions)
    return false_positives/negatives

def roc_curve(labels, prediction_values):
    curve = []
    for param in range(101):
        predictions = prediction_values>param/100.0
        ss_point = [1-specificity(labels, predictions), sensitivity(labels, predictions)]
        curve.append(ss_point)
    return curve

def pr_curve(labels, prediction_values):
    curve = []
    for param in range(101):
        predictions = prediction_values>param/100.0
        pr_point = [recall(labels, predictions), precision(labels, predictions)]
        curve.append(pr_point)
    return curve

def auc(curve):
    x=curve[:,0]
    y=curve[:,1]
    widths = x[1:]-x[:-1]
    heights = (y[1:]+y[:-1])/2
    return widths * heights
