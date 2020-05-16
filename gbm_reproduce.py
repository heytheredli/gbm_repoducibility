
import pandas as pd
import numpy as np
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV

import xgboost as xgb
import lightgbm as lgb

diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target

y = np.where(y>150, 1, 0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=13)

params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01}

reg = ensemble.GradientBoostingClassifier(**params)
reg.fit(X_train, y_train)

auc = roc_auc_score(y_test, reg.predict(X_test))
print(f'sklearn out of the box: {auc}')


D_train = xgb.DMatrix(X_train, label=y_train)
D_test = xgb.DMatrix(X_test, label=y_test)

clf = xgb.XGBClassifier()
parameters = {
     "eta"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
     "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
     "min_child_weight" : [ 1, 3, 5, 7 ],
     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
     }

grid = GridSearchCV(clf,
                    parameters, n_jobs=4,
                    scoring="neg_log_loss",
                    cv=3)

grid.fit(X_train, y_train)

model = xgb.train(grid.best_params_, D_train, 200)


auc = roc_auc_score(y_test, model.predict(D_test))
print(f'xgboost: {auc}')

parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0
}

model = lgb.train(parameters,
                       X_train,
                       valid_sets=X_test,
                       num_boost_round=200,
                       early_stopping_rounds=30)

auc = roc_auc_score(y_test, model.predict(X_test))
print(f'lightgbm: {auc}')
