import time
start_time = time.time()

import numpy as np
import pandas as pd
import os

from sklearn import pipeline, grid_search
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from xgboost import XGBRegressor

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)

np.random.seed(0)

n_folds = 10
verbose = True
shuffle = False

df_train = pd.read_csv(os.path.expanduser('~/Desktop/Home Depot/Output/train.csv'))
df_test = pd.read_csv(os.path.expanduser('~/Desktop/Home Depot/Output/test.csv'))
submission = os.path.expanduser('~/Desktop/Home Depot/Output/submission.csv')

feature_cols = [col for col in df_train.columns if col not in ['relevance','id']]

X = df_train[feature_cols]
X_submission = df_test[feature_cols]
y = df_train['relevance']
test_ids = df_test['id']

if shuffle:
    idx = np.random.permutation(y.size)
    X_train = y[idx]
    y_train = y[idx]

skf = list(StratifiedKFold(y, n_folds))

clfs = [RandomForestRegressor(n_estimators = 1000, n_jobs = -1, random_state = 2016, verbose = 1, max_features=10, max_depth=20),
        ExtraTreesRegressor(n_estimators = 1000, n_jobs = -1, random_state = 2016, verbose = 1, max_features=1, max_depth=100),
        XGBRegressor(learning_rate=0.25, silent=False, objective="reg:linear",
                     nthread=1, gamma=5, min_child_weight=.1, max_delta_step=0,
                     subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0,
                     reg_lambda=1, scale_pos_weight=1,base_score=0.5, seed=0,
                     missing=None, max_depth=1)]

print("Creating train and test sets for blending.")

dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

for j, clf in enumerate(clfs):
    dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
    for i, (train, test) in enumerate(skf):
        X_train = X.ix[train]
        y_train = y.ix[train]
        X_test = X.ix[test]
        y_test = y.ix[test]
        clf.fit(X_train, y_train)
        y_submission = clf.predict(X_test)
        dataset_blend_train[test, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict(X_submission)
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

param_grid = {}

#clf = RandomForestRegressor(n_estimators = 1000, n_jobs = -1, random_state = 2016, verbose = 1)
rf_model = grid_search.GridSearchCV(estimator = clfs, param_grid = param_grid, n_jobs = 1, cv = 10, verbose = 20, scoring=RMSE)
rf_model.fit(X, y)
y_submission = rf_model.predict(X_submission)

print ('')
print ("Best CV score:")
print (rf_model.best_score_)
print (rf_model.best_score_ + 0.47003199274)

with open(submission, "w") as outfile:
    outfile.write("id,relevance\n")
    for e, val in enumerate(list(y_submission)):
        outfile.write("%s,%s\n"%(test_ids[e],val))
