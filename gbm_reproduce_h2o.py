import h2o
h2o.init()


import os
import pandas as pd
import numpy as np
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV

import multiprocessing

# prep data
diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target

X = pd.DataFrame(X)

for col in X.columns.tolist():
    for _ in range(10):
        X[f'{col}_{_}'] = np.random.permutation(X[col].values)

X = np.array(X)

#bc_data = datasets.load_breast_cancer()
#X, y = bc_data.data, bc_data.target

y = np.where(y>150, 1, 0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=13)


df_train = pd.DataFrame(X_train)
df_train['response'] = y_train
df_train = h2o.H2OFrame(df_train)

df_test = pd.DataFrame(X_test)
df_test['response'] = y_test
df_test = h2o.H2OFrame(df_test)

param = {
      "ntrees" : 100
    , "max_depth" : 10
    , "learn_rate" : 0.02
    , "sample_rate" : 0.7
    , "col_sample_rate_per_tree" : 0.9
    , "min_rows" : 5
    , "seed": 42
    , "score_tree_interval": 100
}
from h2o.estimators import H2OXGBoostEstimator
model = H2OXGBoostEstimator(**param)

model.train(y = "response", training_frame = df_train)
prediction = model.predict(df_test)

auc = roc_auc_score(y_test, prediction.as_data_frame())
print(f'h2o auc: {auc}')
print(model.params.get('seed').get('actual'))
print(df_train.head())

