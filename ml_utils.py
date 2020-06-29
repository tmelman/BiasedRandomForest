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
    y = y.values.flatten()
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
    if predicted_positives==0:
        return 0
    return true_positives/predicted_positives

def recall(labels, predictions):
    # true positives/labeled positives
    true_positives = sum(np.logical_and(labels, predictions))
    labeled_positives = sum(labels)
    if labeled_positives==0:
        return 0
    return true_positives/labeled_positives

def sensitivity(labels, predictions):
    return recall(labels, predictions)

def specificity(labels, predictions):
    true_negatives = sum(np.logical_and(1-labels, 1-predictions))
    labeled_negatives = sum(1-labels)
    if labeled_negatives==0:
        return 0
    return true_negatives/labeled_negatives

def roc_curve(labels, prediction_values):
    curve = [[1,1]]
    for p in range(101):
        param = p/100
        predictions = prediction_values>param
        ss_point = [1-specificity(labels, predictions), sensitivity(labels, predictions)]
        curve.append(ss_point)
    return curve

def pr_curve(labels, prediction_values):
    curve = [[1,1]]
    for p in range(101):
        param = p/100
        predictions = prediction_values>param
        pr_point = [recall(labels, predictions), precision(labels, predictions)]
        curve.append(pr_point)
    return curve

def auc(curve):
    sorted_inds = np.argsort(curve[:,0]) # sort along x
    x=curve[sorted_inds,0]
    y=curve[sorted_inds,1]
    widths = x[1:]-x[:-1]
    heights = (y[1:]+y[:-1])/2
    return sum(widths * heights)
