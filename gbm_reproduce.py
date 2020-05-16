import os
import pandas as pd
import numpy as np
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV

import xgboost as xgb
import lightgbm as lgb

import multiprocessing

# prep data
diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target

y = np.where(y>150, 1, 0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=13)

def run_sklearn():
    parameters = {
        'max_depth': [2, 4, 6, 8],
        'min_samples_split': [3, 5, 8, 10],
        'learning_rate': [0.005, 0.01, 0.05, 0.1]
        }

    clf = ensemble.GradientBoostingClassifier()
    grid = GridSearchCV(clf,
                        parameters, n_jobs=4,
                        scoring="neg_log_loss",
                        cv=3)
    grid.fit(X=X_train, y=y_train)

    final_params = grid.best_params_

    reg = ensemble.GradientBoostingClassifier(**final_params)
    reg.fit(X_train, y_train)

    auc = roc_auc_score(y_test, reg.predict(X_test))
    return auc

def run_xgb():
    D_train = xgb.DMatrix(X_train, label=y_train)
    D_test = xgb.DMatrix(X_test, label=y_test)

    clf = xgb.XGBClassifier()
    parameters = {
         "eta"    : [0.05, 0.15, 0.20, 0.30 ] ,
         "max_depth"        : [ 3, 8, 15],
         "min_child_weight" : [ 1, 3, 5, 7 ],
         "gamma"            : [ 0.0, 0.2 , 0.4 ],
         "colsample_bytree" : [ 0.3, 0.5 , 0.7 ]
         }

    grid = GridSearchCV(clf,
                        parameters, n_jobs=4,
                        scoring="neg_log_loss",
                        cv=3)

    grid.fit(X_train, y_train)
    
    final_params = grid.best_params_
    final_params['random_state'] = 42

    model = xgb.train(final_params, D_train, 200)


    auc = roc_auc_score(y_test, model.predict(D_test))
    return auc

def run_lgb():
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    clf = lgb.LGBMClassifier()
    parameters = {
        'num_leaves': [31, 127],
        'reg_alpha': [0.1, 0.5],
        'min_data_in_leaf': [30, 50, 100, 300, 400],
        'lambda_l1': [0, 1, 1.5],
        'lambda_l2': [0, 1],
        }

    grid = GridSearchCV(clf,
                        parameters, n_jobs=4,
                        scoring="neg_log_loss",
                        cv=3)
    grid.fit(X=X_train, y=y_train)
    final_params = grid.best_params_
    final_params['metric'] = 'auc'
    final_params['seed'] = 42

    model = lgb.train(final_params,
                           lgb_train,
                           valid_sets=lgb_eval,
                           num_boost_round=200,
                           early_stopping_rounds=30)

    auc = roc_auc_score(y_test, model.predict(X_test, num_iteration=model.best_iteration))
    return auc

def main():
    print('running sklearn')
    sklearn_auc = run_sklearn()
    print('running xgboost')
    xgb_auc = run_xgb()
    print('running lightgbm')
    lgb_auc = run_lgb()
    
    if os.path.exists('results.txt'):
        with open('results.txt', 'a') as f:
            f.write(f'\nrun results ({multiprocessing.cpu_count()} core(s)): \n')
            f.write(f'sklearn: {sklearn_auc}\n')
            f.write(f'xgboost: {xgb_auc}\n')
            f.write(f'lightgbm: {lgb_auc}\n')
    else:
        with open('results.txt', 'w') as f:
            f.write(f'\nrun results ({multiprocessing.cpu_count()} core(s)): \n')
            f.write(f'sklearn: {sklearn_auc}\n')
            f.write(f'xgboost: {xgb_auc}\n')
            f.write(f'lightgbm: {lgb_auc}\n')

if __name__ == "__main__":
    main()

