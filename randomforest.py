# This is an implementation of random forest.
import attr
import numpy as np
import scipy as scp
import pandas as pd
from typing import Callable, List, TypeVar

@attr.s
class DecisionTree():
    # Single decision tree randomly selecting a subset of the 
    # features at each step, for use building a random forest
    data_x = attr.ib(type=pd.DataFrame, init=True)
    data_y = attr.ib(type=pd.DataFrame, init=True) # binary labels

    min_sample_leaf = attr.ib(type=int, default=-1, init=True)
    max_features = attr.ib(type=Callable, default=np.sqrt, init=True)
    max_depth = attr.ib(type=int, default=-1, init=True)

    left_child = attr.ib(type=TypeVar('DecisionTree'), default=None, init=False)
    right_child = attr.ib(type=TypeVar('DecisionTree'), default=None, init=False)
    feature_split = attr.ib(type=str, default=None, init=False)
    value_split = attr.ib(type=np.float, default=None, init=False)


    def gini(self, labels):
        # Calculate the partial gini index of a branch, given labels
        # 1-(p_pos)^2-(1-p_pos)^2
        p_pos = np.mean(labels)
        return 2*p_pos*(1-p_pos)

    def _get_optimal_split(self):
        # Return the optimal splitting criteria. 
        # 1. select random subset of features:
        num_features = int(self.max_features(len(self.data_x.columns)))
        column_subset = np.random.choice(self.data_x.columns, num_features, replace=False)
        # 2. initialize "optimal split"
        optimal_split = {
            'feature_split': None, 
            'value_split': None, 
            'gini': 999, 
            'left_inds': [], 
            'right_inds': []
        }
        for col in column_subset: # for each random feature
            for value_split in self.data_x[col]: # for each value along that feature
                # a) split the data
                p_left_indices = np.where(self.data_x[col]<value_split)[0]
                p_right_indices = np.where(self.data_x[col]>=value_split)[0]
                left_branch = self.data_y.iloc[p_left_indices].values
                right_branch = self.data_y.iloc[p_right_indices].values
                # b) calculate the gini index of the split
                p_left = len(left_branch)/(len(left_branch)+len(right_branch))
                p_right = 1-p_left
                gini = (
                    self.gini(left_branch)*p_left + 
                    self.gini(right_branch)*p_right
                )
                # c) if it's smaller than previously seen gini's, save it
                gini = 1 if np.isnan(gini) else gini
                if gini < optimal_split['gini']: 
                    optimal_split = {
                        'feature_split': col, 
                        'value_split': value_split, 
                        'gini': gini, 
                        'left_inds': p_left_indices, 
                        'right_inds': p_right_indices
                    }
        if (
            len(optimal_split['left_inds'])<self.min_sample_leaf or
            len(optimal_split['right_inds'])<self.min_sample_leaf
        ): # if not enough data in one of the branches, do not split
            return None
        return optimal_split
    
    def make_tree(self):
        # 0. Base case: Evaluate gini, if 0 (pure leaf) or if max_depth is reached: stop
        if (self.gini(self.data_y.values)==0 or 
            self.max_depth==0 or 
            self.min_sample_leaf==len(self.data_x)):
            return
        # 1. Get optimal split
        split = self._get_optimal_split()
        if split is None:
            return
        # 2. Populate "feature_split", "value_split", update "max_depth"
        self.feature_split = split['feature_split']
        self.value_split = split['value_split']
        self.max_depth -= 1
        left_inds = split['left_inds']
        right_inds = split['right_inds']
        left_data_x, right_data_x = self.data_x.iloc[left_inds], self.data_x.iloc[right_inds]
        left_data_y, right_data_y = self.data_y.iloc[left_inds], self.data_y.iloc[right_inds]
        # 3. Delete data from parent node to avoid memory blow-up
        self.data_x=None
        self.data_y=None
        # 4. Make child trees
        self.left_child = DecisionTree(
            data_x=left_data_x, 
            data_y=left_data_y, 
            min_sample_leaf=self.min_sample_leaf,
            max_depth=self.max_depth-1,
            max_features=self.max_features
        )
        self.right_child = DecisionTree(
            data_x=right_data_x, 
            data_y=right_data_y, 
            min_sample_leaf=self.min_sample_leaf,
            max_depth=self.max_depth-1,
            max_features=self.max_features
        )
        self.left_child.make_tree()
        self.right_child.make_tree()

            

    def predict(self, new_data:pd.DataFrame):
        # Precdict new_data, a 1D dataframe
        # Check if the feature feature_split is greater than or less than value_split
        # If left/right child exists then go to that branch, call predict() on the child
        if self.left_child and self.right_child:
            if new_data[self.feature_split]<self.value_split:
                return self.left_child.predict(new_data)
            else:
                return self.right_child.predict(new_data)
        # else return consensus of leaf
        else:
            return np.mean(self.data_y.values)
        
@attr.s
class RandomForest():
    # Creates a random forest classifier of DecisionTrees with randomly subsetted features
    data_x = attr.ib(type=pd.DataFrame, init=True)
    data_y = attr.ib(type=pd.DataFrame, init=True) # binary labels
    hyper_params = attr.ib(
        type=dict, default= {
            'max_depth': 100,
            'max_features': np.sqrt,
            'min_sample_leaf': 5,
            'ntrees':100,
        },
        init=True
    )
    trees = attr.ib(default=[], type=List[DecisionTree],init=False)
    
    def make_forest(self):
        # Make the 'ntrees' DecisionTrees
        ntrees = self.hyper_params['ntrees']
        for n in range(ntrees):
            if (n+1)%10==0:
                print(f'Making tree #{n+1}',end='\r')
            new_tree = DecisionTree(
                data_x=self.data_x,
                data_y=self.data_y,
                max_features=self.hyper_params['max_features'],
                min_sample_leaf=self.hyper_params['min_sample_leaf'],

            )
            new_tree.make_tree()
            self.trees.append(new_tree)

    def predict(self, new_data: pd.DataFrame):
        # Get mean prediction (not an integer)
        predictions = []
        for tree in self.trees:
            predictions.append(np.round(tree.predict(new_data)))
        
        return np.mean(predictions)

@attr.s
class BiasedRandomForest():
    # Implementation of the Bader-El-Den BRAF algorithm.
    # Makes two Random Forests, one with a subset of the data.
    data_x = attr.ib(type=pd.DataFrame, init=True)
    data_y = attr.ib(type=pd.DataFrame, init=True)
    hyper_params = attr.ib(
        type=dict, default= {
            'max_depth': 100,
            'max_features': np.sqrt,
            'min_sample_leaf': 5,
            'k': 10,
            'p': .5,
            's': 100,
        },
        init=True
    )
    rf_all = attr.ib(type=RandomForest, init=False, default=None)
    rf_knn = attr.ib(type=RandomForest, init=False, default=None)

    def get_knn_Maj_from_Min(self, k=10):
        # Crux of the algorithm to undersample majority class
        # 1. identify majority class
        yvals = self.data_y.values.flatten()
        majority_class = np.round(np.mean(yvals))
        min_inds = np.where(yvals!=majority_class)[0]
        maj_inds = np.where(yvals==majority_class)[0]
        min_set_x = self.data_x.iloc[min_inds]
        maj_set_x = self.data_x.iloc[maj_inds]
        min_set_y = self.data_y.iloc[min_inds]
        maj_set_y = self.data_y.iloc[maj_inds]
        # 2. get distance of majority class members from minority class members
        dist_mat = scp.spatial.distance_matrix(min_set_x.values, maj_set_x.values) 
        # 3. get the k nearest neighbors to each minority class member
        knn = np.unique(np.array([np.argsort(i)[:k] for i in dist_mat]).flatten())
        maj_set_knn_x = maj_set_x.iloc[knn]
        maj_set_knn_y = maj_set_y.iloc[knn]
        return min_set_x.append(maj_set_knn_x), min_set_y.append(maj_set_knn_y) 

    def make_rfs(self):
        # Make the two random forests
        # 1. Undersample the majority class (per Bader-El-Din)
        knn_subset_x, knn_subset_y = self.get_knn_Maj_from_Min(self.hyper_params['k'])
        # 2. Make two forests trained on different data
        hparams = {
            'max_depth': 100,
            'max_features': np.sqrt,
            'min_sample_leaf': 2,
            'ntrees': int(self.hyper_params['p']*self.hyper_params['s'])
        }
        self.rf_all = RandomForest(data_x=self.data_x, data_y=self.data_y, hyper_params=hparams)
        self.rf_all.make_forest()
        hparams['ntrees']= int((1-self.hyper_params['p'])*self.hyper_params['s'])
        self.rf_knn = RandomForest(data_x=knn_subset_x, data_y=knn_subset_y, hyper_params=hparams)
        self.rf_knn.make_forest()

    def predict(self, new_data: pd.DataFrame):
        # Predict label of a new data point
        pred_all = self.rf_all.predict(new_data)
        pred_knn = self.rf_knn.predict(new_data)
        # return a prediction weighted by p
        return (
            pred_all*self.hyper_params['p'] +
            pred_knn*(1-self.hyper_params['p'])
        )


