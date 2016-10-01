import warnings; warnings.filterwarnings("ignore")
import time
start_time = time.time()

import pandas as pd
import os

from sklearn import pipeline, grid_search
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer

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

df_all = pd.read_csv(os.path.expanduser('~/Desktop/Home Depot/Features/feature_array.csv'))
feature_cols = [col for col in df_all.columns if col not in ['Unnamed: 0']]
df_all = df_all[feature_cols]

df_train = pd.read_csv('~/Desktop/Home Depot/Data/train.csv', encoding="ISO-8859-1")#[:1000]
num_train = df_train.shape[0]

df_train = df_all[:num_train]
df_test = df_all[num_train:]

y_train = df_train['relevance']
id_test = df_test['id']
X_train = df_train[:]
X_test = df_test[:]

if __name__ == '__main__':

    rfr = RandomForestRegressor(n_estimators = 2000, n_jobs = -1, random_state = 2016, verbose = 1)
    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
    tsvd = TruncatedSVD(n_components=10, random_state = 2016)
    clf = pipeline.Pipeline([
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
                            'txt3': 0.0,
                            'txt4': 0.5,

                            },
                    n_jobs = -1
                    )),
            ('rfr', rfr)])
    param_grid = {'rfr__max_features': [10], 'rfr__max_depth': [20]}
    model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = -1, cv = 2, verbose = 20, scoring=RMSE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Best parameters found by grid search:")
    print(model.best_params_)
    print("Best CV score:")
    print(model.best_score_)
    print(model.best_score_ + 0.47003199274)

    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(os.path.expanduser('~/Desktop/Home Depot/Output/rf_submission.csv'),index=False)
    print("--- Training & Testing: %s minutes ---" % round(((time.time() - start_time)/60),2))
