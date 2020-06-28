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
    indices_left = np.array(range(len(y)))
    fold_membership = np.zeros(len(y))-1
    for i in range(k-1):
        y_unassigned = np.array(range(len(y)))[fold_membership==-1]
        inds_ith_fold, _ = get_stratified_train_test_split(y[y_unassigned], n_each_fold)
        fold_membership[y_unassigned[inds_ith_fold]] = i
    fold_membership[fold_membership==-1] = k-1
    return fold_membership


def get_knn_A_from_B(setA, setB, k):
    dist_mat = scp.spatial.distance_matrix(setA, setB)
    knn = np.unique(np.array([np.argsort(i)[:k] for i in dist_mat]).flatten())
    return setB[knn,:]
