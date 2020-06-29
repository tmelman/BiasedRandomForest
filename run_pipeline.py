# imports
import pandas as pd
from randomforest import BiasedRandomForest
from ml_utils import (
    precision,
    recall,
    roc_curve,
    precision
    auc,
)

# load PIMA data after pre-processing
pima = pd.open_csv('../diabetes_processed.csv')


# create BRAF, 10fold cval


# predict

# print out precision, recall, ROC-AUC, PR-AUC

# save images for ROC, PR