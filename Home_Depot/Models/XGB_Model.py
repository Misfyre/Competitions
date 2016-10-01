import warnings; warnings.filterwarnings("ignore")
import time
start_time = time.time()

import pandas as pd
import numpy as np
import xgboost as xgb
import os

import operator
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import pipeline, grid_search

import random
random.seed(2016)

df_all = pd.read_csv(os.path.expanduser('~/Desktop/Home Depot/Features/feature_array.csv'))
feature_cols = [col for col in df_all.columns if col not in ['Unnamed: 0']]
df_all = df_all[feature_cols]

df_train = pd.read_csv('~/Desktop/Home Depot/Data/train.csv', encoding="ISO-8859-1")#[:1000]
num_train = df_train.shape[0]

df_train = df_all[:num_train]
df_test = df_all[num_train:]

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

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_
def feature_to_coords(list):
    coords = []
    count = 1
    for x in list:
        x = count
        count += 1
        coords.append(x)
    return coords
def xgb_importance(clf):
    clf.fit(X_train, y_train)
    labeldummy, estimator = clf.steps[-1]
    importances = estimator.booster().get_fscore()
    coordinates = (list(zip(X_train.columns, importances.values())))
    coordinates = sorted(coordinates, key=lambda x: x[1], reverse = True)
    return coordinates
def fix_outliers(input):
    if input > 3:
        input = 3.0
    elif input < 1:
        input = 1.0
    return input

print ('')

RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)

y_train = df_train['relevance']
id_test = df_test['id']
X_train = df_train[:]
X_test = df_test[:]

xgr_1 = xgb.XGBRegressor(n_estimators=3000, max_depth=5, seed=2016, missing=np.nan, learning_rate=0.02, subsample=0.9,
                         colsample_bytree=0.85, objective='reg:linear', silent = True)

param_grid = {}
id_results = id_test[:]
id_results_1 = id_test[:]
id_results_2 = id_test[:]
id_results_3 = id_test[:]

tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
tsvd = TruncatedSVD(n_components=95, random_state = 2016)

clf_3 = pipeline.Pipeline([
            ('union', FeatureUnion(
                        transformer_list = [
                            ('cst',  cust_regression_vals()),
                            ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                            ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                            ('txt3', pipeline.Pipeline([('s3', cust_txt_col(key='product_description')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
                            ('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)])),
                            ],
                        transformer_weights = {
                            'cst': 1.0,
                            'txt1': 0.5,
                            'txt2': 0.25,
                            'txt3': 0.25,
                            'txt4': 0.25,
                            },
                    n_jobs = -1
                    )),
            ('xgr_1', xgr_1)])

xgr_model_1 = grid_search.GridSearchCV(estimator = clf_3, param_grid = {}, n_jobs = -1, cv = 2, verbose = 0, scoring=RMSE)
xgr_model_1.fit(X_train, y_train)
y_pred = xgr_model_1.predict(X_test)
df_in = pd.DataFrame({'id': id_test, 'relevance': y_pred})
id_results = pd.concat([id_results, df_in['relevance']], axis = 1)
id_results_1 = pd.concat([id_results_1, df_in['relevance']], axis = 1)

for i in range(len(y_pred)):
    if y_pred[i]<1.0:
        y_pred[i] = 1.0
    if y_pred[i]>3.0:
        y_pred[i] = 3.0

df_in = df_in[['id', 'relevance']]
df_in.columns = ['id', 'relevance']
df_in.to_csv(os.path.expanduser('~/Desktop/Home Depot/Output/Models/xgr_submission_1.csv'), index=False)

xgb_feature_importance = xgb_importance(clf_3)
print (xgb_feature_importance)
xgb_1, xgb_2, xgb_3, xgb_4, xgb_5 = ((list(zip(*xgb_feature_importance)))[0])[-5:]

class cust_regression_xgb_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        d_col_drops=['id','relevance','search_term','product_title','product_description','product_info','attr','brand',
                     xgb_1, xgb_2, xgb_3, xgb_4, xgb_5]
        hd_searches = hd_searches.drop(d_col_drops,axis=1).values
        return hd_searches

print ("XGR CV:", xgr_model_1.best_score_, '||', "Best Params:", xgr_model_1.best_params_, '||', "Time: ", round(((time.time() - start_time)/60),2))

clf_3 = pipeline.Pipeline([
            ('union', FeatureUnion(
                        transformer_list = [
                            ('cst',  cust_regression_xgb_vals()),
                            ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                            ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                            ('txt3', pipeline.Pipeline([('s3', cust_txt_col(key='product_description')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
                            ('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)])),
                            ],
                        transformer_weights = {
                            'cst': 1.0,
                            'txt1': 0.5,
                            'txt2': 0.25,
                            'txt3': 0.25,
                            'txt4': 0.25,
                            },
                    n_jobs = -1
                    )),
            ('xgr', xgr_1)])

xgr_model_2 = grid_search.GridSearchCV(estimator = clf_3, param_grid = {}, n_jobs = -1, cv = 2, verbose = 0, scoring=RMSE)
xgr_model_2.fit(X_train, y_train)
y_pred = xgr_model_2.predict(X_test)
df_in = pd.DataFrame({'id': id_test, 'relevance': y_pred})
id_results = pd.concat([id_results, df_in['relevance']], axis = 1)
id_results_2 = pd.concat([id_results_2, df_in['relevance']], axis = 1)

for i in range(len(y_pred)):
    if y_pred[i]<1.0:
        y_pred[i] = 1.0
    if y_pred[i]>3.0:
        y_pred[i] = 3.0

df_in = df_in[['id', 'relevance']]
df_in.columns = ['id', 'relevance']
df_in.to_csv(os.path.expanduser('~/Desktop/Home Depot/Output/Models/xgr_submission_2.csv'), index=False)

xgb_feature_importance_1 = xgb_importance(clf_3)
print (xgb_feature_importance_1)
xgb_1, xgb_2, xgb_3, xgb_4, xgb_5 = ((list(zip(*xgb_feature_importance_1)))[0])[-5:]

class cust_regression_xgb_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        d_col_drops=['id','relevance','search_term','product_title','product_description','product_info','attr','brand',
                     xgb_1, xgb_2, xgb_3, xgb_4, xgb_5]
        hd_searches = hd_searches.drop(d_col_drops,axis=1).values
        return hd_searches

print ("XGR CV:", xgr_model_2.best_score_, '||', "Best Params:", xgr_model_2.best_params_, '||', "Time: ", round(((time.time() - start_time)/60),2))

clf_3 = pipeline.Pipeline([
            ('union', FeatureUnion(
                        transformer_list = [
                            ('cst',  cust_regression_xgb_vals()),
                            ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                            ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                            ('txt3', pipeline.Pipeline([('s3', cust_txt_col(key='product_description')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
                            ('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)])),
                            ],
                        transformer_weights = {
                            'cst': 1.0,
                            'txt1': 0.5,
                            'txt2': 0.25,
                            'txt3': 0.25,
                            'txt4': 0.25,
                            },
                    n_jobs = -1
                    )),
            ('xgr', xgr_1)])

xgr_model_3 = grid_search.GridSearchCV(estimator = clf_3, param_grid = {}, n_jobs = -1, cv = 2, verbose = 0, scoring=RMSE)
xgr_model_3.fit(X_train, y_train)
y_pred = xgr_model_3.predict(X_test)
df_in = pd.DataFrame({'id': id_test, 'relevance': y_pred})
id_results = pd.concat([id_results, df_in['relevance']], axis = 1)
id_results_3 = pd.concat([id_results_3, df_in['relevance']], axis = 1)

for i in range(len(y_pred)):
    if y_pred[i]<1.0:
        y_pred[i] = 1.0
    if y_pred[i]>3.0:
        y_pred[i] = 3.0

df_in = df_in[['id', 'relevance']]
df_in.columns = ['id', 'relevance']
df_in.to_csv(os.path.expanduser('~/Desktop/Home Depot/Output/Models/xgr_submission_3.csv'), index=False)

xgb_feature_importance_2 = xgb_importance(clf_3)
print (xgb_feature_importance_2)
xgb_1, xgb_2, xgb_3, xgb_4, xgb_5 = ((list(zip(*xgb_feature_importance_1)))[0])[-5:]

class cust_regression_xgb_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        d_col_drops=['id','relevance','search_term','product_title','product_description','product_info','attr','brand',
                     xgb_1, xgb_2, xgb_3, xgb_4, xgb_5]
        hd_searches = hd_searches.drop(d_col_drops,axis=1).values
        return hd_searches

print ("XGR CV:", xgr_model_3.best_score_, '||', "Best Params:", xgr_model_3.best_params_, '||', "Time: ", round(((time.time() - start_time)/60),2))

print ('')

print ("XGB CV Model 1", xgr_model_1.best_score_)
print ("XGB CV Model 2", xgr_model_2.best_score_)
print ("XGB CV Model 3", xgr_model_3.best_score_)

print ('')

print ("XGB CV Difference (1-2)", xgr_model_2.best_score_ - xgr_model_1.best_score_)
print ("XGB CV Difference (2-3)", xgr_model_3.best_score_ - xgr_model_2.best_score_)
print ("XGB CV Difference (1-3)", xgr_model_3.best_score_ - xgr_model_1.best_score_)

print ('')

id_results['avg'] = id_results.drop('id', axis = 1).apply(np.average, axis = 1)
id_results['min'] = id_results.drop('id', axis = 1).apply(min, axis = 1)
id_results['max'] = id_results.drop('id', axis = 1).apply(max, axis = 1)
id_results['diff'] = id_results['max'] - id_results['min']

id_results_1['avg'] = id_results_1.drop('id', axis = 1).apply(np.average, axis = 1)
id_results_1['min'] = id_results_1.drop('id', axis = 1).apply(min, axis = 1)
id_results_1['max'] = id_results_1.drop('id', axis = 1).apply(max, axis = 1)
id_results_1['diff'] = id_results_1['max'] - id_results['min']

id_results_2['avg'] = id_results_2.drop('id', axis = 1).apply(np.average, axis = 1)
id_results_2['min'] = id_results_2.drop('id', axis = 1).apply(min, axis = 1)
id_results_2['max'] = id_results_2.drop('id', axis = 1).apply(max, axis = 1)
id_results_2['diff'] = id_results_2['max'] - id_results['min']

id_results.to_csv(os.path.expanduser('~/Desktop/Home Depot/Output/Analysis/overall_ensemble.csv'), index=False)
id_results_1.to_csv(os.path.expanduser('~/Desktop/Home Depot/Output/Analysis/model_1_ensemble.csv'), index=False)
id_results_2.to_csv(os.path.expanduser('~/Desktop/Home Depot/Output/Analysis/model_2_ensemble.csv'), index=False)

analysis = id_results[['avg', 'min', 'max', 'diff']]
analysis_1 = id_results_1[['avg', 'min', 'max', 'diff']]
analysis_2 = id_results_2[['avg', 'min', 'max', 'diff']]

print (analysis_1[:10])

print ('')
print ('---------------------------------------------')
print ('')
print (analysis_2[:10])

print ('')
print ('---------------------------------------------')
print ('')

print (analysis[:10])

ds = id_results[['id', 'avg']]
ds.columns = ['id', 'relevance']
ds['relevance'] = ds['relevance'].map(lambda x:fix_outliers(x))
ds.to_csv(os.path.expanduser('~/Desktop/Home Depot/Output/submission.csv'), index=False)

ds_1 = id_results_1[['id', 'avg']]
ds_1.columns = ['id', 'relevance']
ds_1['relevance'] = ds_1['relevance'].map(lambda x:fix_outliers(x))
ds_1.to_csv(os.path.expanduser('~/Desktop/Home Depot/Output/submission_1.csv'), index=False)

ds_2 = id_results_2[['id', 'avg']]
ds_2.columns = ['id', 'relevance']
ds_2['relevance'] = ds_2['relevance'].map(lambda x:fix_outliers(x))
ds_2.to_csv(os.path.expanduser('~/Desktop/Home Depot/Output/submission_2.csv'), index=False)

x_coords = list(np.asarray(id_results['id']))
y_coords = list(np.asarray(id_results['avg']))
x_coords_1 = list(np.asarray(id_results_1['id']))
y_coords_1 = list(np.asarray(id_results_1['avg']))
x_coords_2 = list(np.asarray(id_results_2['id']))
y_coords_2 = list(np.asarray(id_results_2['avg']))

