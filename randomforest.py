# This is an implementation of random forest.
import attr
import numpy as np
import pandas as pd
from typing import Callable, List, TypeVar

@attr.s
class DecisionTree():
    data_x = attr.ib(type=pd.DataFrame, init=True)
    data_y = attr.ib(type=pd.DataFrame, init=True) # binary labels

    min_sample_leaf = attr.ib(type=int, default=-1, init=True)
    max_features = attr.ib(type=Callable, default=np.sqrt, init=True)
    max_depth = attr.ib(type=int, default=-1, init=True)

    left_child = attr.ib(type=TypeVar('DecisionTree'), default=None, init=False)#type=DecisionTree, 
    right_child = attr.ib(type=TypeVar('DecisionTree'), default=None, init=False)#type=DecisionTree, 
    feature_split = attr.ib(type=str, default=None, init=False)
    value_split = attr.ib(type=np.float, default=None, init=False)


    def gini(self, labels):
        # Calculate the partial gini index of a branch, given labels
        p_pos = np.mean(labels)
        return 2*p_pos*(1-p_pos)

    def _get_optimal_split(self):
        # Return the optimal splitting criteria
        results_columns = ['feature_split', 'value_split', 'gini','left_inds','right_inds']
        splits = pd.DataFrame(columns=results_columns)
        num_features = int(self.max_features(len(self.data_x.columns)))
        column_subset = np.random.choice(self.data_x.columns, num_features)
        for col in column_subset:
            for value_split in self.data_x[col]:
                p_left_indices = np.where(self.data_x[col]<value_split)[0]
                p_right_indices = np.where(self.data_x[col]>=value_split)[0]
                
                gini = (
                    self.gini(self.data_y.iloc[p_left_indices].values) + 
                    self.gini(self.data_y.iloc[p_right_indices].values)
                    
                ) # Double check Gini calculation
                gini = 1 if np.isnan(gini) else gini
                results_data = [[col, value_split, gini, p_left_indices, p_right_indices]]
                results_df = pd.DataFrame(data=results_data,columns=results_columns)
                splits = splits.append(results_df)
        optimal_split = splits.iloc[np.where(splits.gini.values==np.min(splits.gini.values))[0]]
        if (
            len(optimal_split.left_inds)<self.min_sample_leaf or
            len(optimal_split.right_inds)<self.min_sample_leaf
        ):
            return None
        return optimal_split
    
    def make_tree(self):
        # Evaluate gini, if 0 or if max_depth is reached: stop
        if (self.gini(self.data_y.values)==0 or 
            self.max_depth==0 or 
            self.min_sample_leaf==len(self.data_x)):
            return
        # Else, split on criterion, populate "feature_split", "value_split", and update "max_depth" (--)
        # Delete data to avoid memory blow-up, move to children
        # Then call "make_tree" on children
        else:
            split = self._get_optimal_split()
            if split is None:
                return
            self.feature_split = split.feature_split.iloc[0]
            self.value_split = split.value_split.iloc[0]
            left_inds = split.left_inds.iloc[0]
            right_inds = split.right_inds.iloc[0]
            left_data_x, right_data_x = self.data_x.iloc[left_inds], self.data_x.iloc[right_inds]
            left_data_y, right_data_y = self.data_y.iloc[left_inds], self.data_y.iloc[right_inds]
            self.data_x=None
            self.data_y=None
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
            return np.round(np.mean(self.data_y.values))
        
@attr.s
class RandomForest():
    data_x = attr.ib(type=pd.DataFrame, init=True)
    data_y = attr.ib(type=pd.DataFrame, init=True) # binary labels
    hyper_params = attr.ib(
        type=dict, default= {
            'max_depth': 100,
            'max_features': np.sqrt,
            'min_sample_leaf': 2,
            'ntrees':100,
        },
        init=True
    )
    trees = attr.ib(default=[], type=List[DecisionTree],init=False)
    
    def make_forest(self):
        ntrees = self.hyper_params['ntrees']
        for n in range(ntrees):
            new_tree = DecisionTree(
                data_x=self.data_x,
                data_y=self.data_y,
                max_features=self.hyper_params['max_features'],
                min_sample_leaf=self.hyper_params['min_sample_leaf'],

            )
            new_tree.make_tree()
            self.trees.append(new_tree)

    def predict(self, new_data: pd.DataFrame):
        predictions = []
        for tree in self.trees:
            predictions.append([tree.predict(new_data)])
        
        return np.mean(predictions)

