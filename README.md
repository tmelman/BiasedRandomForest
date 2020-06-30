Code for implementation of BRAF/Biased Random Forest, from Bader-El-Den 2019 (IEEE). 

To run the pipeline on the PIMA dataset, run `python run_pipeline.py` from the command line.

Credit Tamar Melman, 2020

This code contains 3 files: 
* ml_utils.py: utility functions to calculate metrics of interest for ML algorithm evaluation
* randomforest.py: script defining DecisionTree, RandomForest, and BiasedRandomForest implementations
* run_pipeline.py, which runs the entire analysis pipeline to train the model and output metrics.

BiasedRandomForest is for demonstration purposes and is not recommended for ML applications; for imbalanced data, I would recommend one of the following approaches:
* Use a weighted Random Forest
* modifying the algorithm to pick a balanced subset of the majority class
* using SMOTE upsampling with a standard RandomForest



