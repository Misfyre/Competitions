print ('')
print ("Initializing Libraries...")

import pandas as pd
import numpy as np
import sys
import os

import xgboost as xgb
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

#---------------------------------------------------------Data---------------------------------------------------------#

def convert_data():

    seed = 7
    np.random.seed(seed)

    print ('')
    print ("### ----- PART 1 ----- ###")

    print ("# Read App Events")
    app_events = pd.read_csv(os.path.expanduser('~/Desktop/TalkingData/Data/app_events.csv'), dtype={'device_id' : np.str})
    app_events= app_events.groupby("event_id")["app_id"].apply(lambda x: " ".join(set("app_id:" + str(s) for s in x)))

    print ("# Read Events")
    events = pd.read_csv(os.path.expanduser('~/Desktop/TalkingData/Data/events.csv'), dtype={'device_id': np.str})
    events["app_id"] = events["event_id"].map(app_events)
    events = events.dropna()
    del app_events

    events = events[["device_id", "app_id"]]
    events.info()

    events.loc[:,"device_id"].value_counts(ascending=True)
    events = events.groupby("device_id")["app_id"].apply(lambda x: " ".join(set(str(" ".join(str(s) for s in x)).split(" "))))
    events = events.reset_index(name="app_id")

    events = pd.concat([pd.Series(row['device_id'], row['app_id'].split(' ')) for _, row in events.iterrows()]).reset_index()
    events.columns = ['app_id', 'device_id']
    f3 = events[["device_id", "app_id"]]

    print ('')
    print ("### ----- PART 2 ----- ###")

    print ("# Read App Labels")
    app_labels = pd.read_csv(os.path.expanduser('~/Desktop/TalkingData/Data/app_labels.csv'))
    label_cat = pd.read_csv(os.path.expanduser('~/Desktop/TalkingData/Data/label_categories.csv'))
    label_cat=label_cat[['label_id','category']]

    app_labels=app_labels.merge(label_cat,on='label_id',how='left')
    app_labels = app_labels.groupby(["app_id","category"]).agg('size').reset_index()
    app_labels = app_labels[['app_id','category']]
    print ("# App Labels Done")

    print ("## Handling Events Data for Merging with App Labels")
    events['app_id'] = events['app_id'].map(lambda x : x.lstrip('app_id:'))
    events['app_id'] = events['app_id'].astype(str)
    app_labels['app_id'] = app_labels['app_id'].astype(str)

    print ("## Merging")
    events= pd.merge(events, app_labels, on = 'app_id',how='left').astype(str)

    print ("# Expanding to Multiple Rows")
    events= events.groupby(["device_id","category"]).agg('size').reset_index()
    events= events[['device_id','category']]

    print ("# App Labels Done")
    f5 = events[["device_id", "category"]]

    print ('')
    print ("### ----- PART 3 ----- ###")

    print ("# Read Phone Brand")
    pbd = pd.read_csv(os.path.expanduser('~/Desktop/TalkingData/Data/phone_brand_device_model.csv'), dtype={'device_id': np.str})
    pbd.drop_duplicates('device_id', keep='first', inplace=True)

    print ("# Generate Train and Test")
    train = pd.read_csv(os.path.expanduser('~/Desktop/TalkingData/Data/gender_age_train.csv'), dtype={'device_id': np.str})
    test = pd.read_csv(os.path.expanduser('~/Desktop/TalkingData/Data/gender_age_test.csv'), dtype={'device_id': np.str})
    train.drop(["age", "gender"], axis=1, inplace=True)
    test["group"] = np.nan

    Y = train["group"]
    label_group = LabelEncoder()
    Y = label_group.fit_transform(Y)
    device_id = test["device_id"]

    Df = pd.concat((train, test), axis=0, ignore_index=True)

    print ('')
    print ("### ----- PART 4 ----- ###")

    Df = pd.merge(Df, pbd, how="left", on="device_id")
    Df["phone_brand"] = Df["phone_brand"].apply(lambda x: "phone_brand:" + str(x))
    Df["device_model"] = Df["device_model"].apply(lambda x: "device_model:" + str(x))

    print ("# Concat All Features")
    f1 = Df[["device_id", "phone_brand"]]   # phone_brand
    f2 = Df[["device_id", "device_model"]]  # device_model
    Df = None

    f1.columns.values[1] = "feature"
    f2.columns.values[1] = "feature"
    f5.columns.values[1] = "feature"
    f3.columns.values[1] = "feature"

    FLS = pd.concat((f1, f2, f3, f5), axis=0, ignore_index=True)

    print ("# User-Item-Feature")
    device_ids = FLS["device_id"].unique()
    feature_cs = FLS["feature"].unique()

    data = np.ones(len(FLS))
    len(data)

    dec = LabelEncoder().fit(FLS["device_id"])
    row = dec.transform(FLS["device_id"])
    col = LabelEncoder().fit_transform(FLS["feature"])
    sparse_matrix = sparse.csr_matrix(
        (data, (row, col)), shape=(len(device_ids), len(feature_cs)))
    sys.getsizeof(sparse_matrix)

    sparse_matrix = sparse_matrix[:, sparse_matrix.getnnz(0) > 0]
    print ("# Sparse Matrix Done")

    del FLS
    del data

    print ("# Split data")
    train_row = dec.transform(train["device_id"])
    train_sp = sparse_matrix[train_row, :]
    test_row = dec.transform(test["device_id"])
    test_sp = sparse_matrix[test_row, :]

    X_train, X_val, y_train, y_val = train_test_split( train_sp, Y, train_size=0.999, random_state=10)

    print ("# Feature Selection")
    print ("# Num of Features: ", X_train.shape[1])
    return X_train, X_val, y_train, y_val, test_sp, label_group, device_id

def run_single(X_train, X_val, y_train, y_val, test_sp, random_state = 0):

    params = {
        "objective": "multi:softprob",
        "num_class": 12,
        "booster": "gblinear",
        "max_depth": 6,
        "eval_metric": "mlogloss",
        "eta": 0.07,
        "silent": 1,
        "alpha": 3
    }

    num_boost_round = 1000
    early_stopping_rounds = 50

    print ('')
    print ("### ----- PART 5 ----- ###")

    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_val, y_val)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
    check = gbm.predict(xgb.DMatrix(X_val))
    score = log_loss(y_val, check)

    print('Check Error Value: {:.6f}'.format(score))

    test_prediction = gbm.predict(xgb.DMatrix(test_sp))
    return test_prediction.tolist(), score

if __name__ == '__main__':

    X_train, X_val, y_train, y_val, test_sp, label_group, device_id = convert_data()
    preds, score = run_single(X_train, X_val, y_train, y_val, test_sp, random_state=0)

    final_submission = pd.DataFrame(preds, index = device_id, columns = label_group.classes_)
    final_submission.to_csv(os.path.expanduser('~/Desktop/TalkingData/Gen1/xgboost.csv'))