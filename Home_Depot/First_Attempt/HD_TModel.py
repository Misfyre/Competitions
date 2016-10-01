import time
start_time = time.time()

import pandas as pd
import os
import numpy as np

from sklearn import grid_search
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, make_scorer
from xgboost import XGBRegressor
from sklearn.cross_validation import StratifiedKFold

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        d_col_drops=['id','relevance','search_term','product_title','product_description','product_info','attr','brand']
        hd_searches = hd_searches.drop(d_col_drops,axis=1).values
        return hd_searches

class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key].apply(str)

RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)

df_train = pd.read_csv(os.path.expanduser('~/Desktop/Home Depot/Output/train.csv'))
df_test = pd.read_csv(os.path.expanduser('~/Desktop/Home Depot/Output/test.csv'))

feature_cols = [col for col in df_train.columns if col not in ['id','relevance','search_term','product_title',
                                                               'product_description','product_info','attr','brand']]

X = df_train[feature_cols]
X_submission = df_test[feature_cols]
y = df_train['relevance']
id_test = df_test['id']

np.random.seed(0)

n_folds = 10
verbose = True
shuffle = False

skf = list(StratifiedKFold(y, n_folds))

if shuffle:
    idx = np.random.permutation(y.size)
    X = X[idx]
    y = y[idx]

clfs = [RandomForestRegressor(n_estimators = 1000, n_jobs = -1, random_state = 2016, verbose = 1, max_depth = 3),
        ExtraTreesRegressor(n_estimators = 1000, n_jobs = -1, random_state = 2016, verbose = 1, max_depth = 3),
        XGBRegressor(learning_rate=0.25, silent=False, objective="reg:linear", nthread=-1, gamma=0, min_child_weight=1,
                                 max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1,
                                 scale_pos_weight=1, base_score=0.5, seed=0, missing=None),
        GradientBoostingRegressor(n_estimators = 1000, random_state = 2016, verbose = 1, max_depth = 3),
        AdaBoostRegressor(n_estimators = 1000, random_state = 2016 )]

dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

for j, clf in enumerate(clfs):
    print (j, clf)
    dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
    for i, (train, test) in enumerate(skf):
        print ("Fold", i)
        X_train = X.ix[train]
        y_train = y.ix[train]
        X_test = X.ix[test]
        y_test = y.ix[test]
        clf.fit(X_train, y_train)
        y_submission = clf.predict(X_test)
        dataset_blend_train[test, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict(X_submission)
    dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

print ("")
print ("Blending.")

rfr = RandomForestRegressor(n_estimators = 1000, n_jobs = -1, random_state = 2016, verbose = 1)
param_grid = {'max_depth': [3]}
rf_model = grid_search.GridSearchCV(estimator = rfr, param_grid = param_grid, n_jobs = 1, cv = 5, verbose = 20, scoring=RMSE)
rf_model.fit(dataset_blend_train, y)
y_submission = rf_model.predict(dataset_blend_test)

print("Best parameters found by Grid Search:")
print(rf_model.best_params_)
print("Best CV score:")
print(rf_model.best_score_)
print(rf_model.best_score_ + 0.47003199274)

pd.DataFrame({"id": id_test, "relevance": y_submission}).to_csv(os.path.expanduser('~/Desktop/Home Depot/Output/Final Submissions/y_submission.csv'),index=False)

