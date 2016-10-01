__author__ = "Nick Sarris (ngs5st)"

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import operator

from xgboost import XGBClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, roc_auc_score

print ('')

def stations_used(row):

    value_counter = 0
    try:
        for value in row:
            if np.isnan(value) == True:
                continue
            else:
                value_counter += 1
    except TypeError:
        if np.isnan(row) == True:
            value_counter = 0
        else:
            value_counter = 1
    return value_counter

def stations_unused(row):

    value_counter = 0
    try:
        for value in row:
            if np.isnan(value) == True:
                value_counter += 1
            else:
                continue
    except TypeError:
        if np.isnan(row) == True:
            value_counter = 1
        else:
            value_counter = 0
    return value_counter

def seperate_levels(train):

    level_0 = []
    level_1 = []
    level_2 = []
    level_3 = []

    for column in train.columns:
        if '_used' not in column:
            if 'L0_' in column:
                list.append(level_0, column)
            elif 'L1_' in column:
                list.append(level_1, column)
            elif 'L2' in column:
                list.append(level_2, column)
            else:
                list.append(level_3, column)

    train_l0 = train[level_0]
    train_l1 = train[level_1]
    train_l2 = train[level_2]
    train_l3 = train[level_3]

    return train_l0, train_l1, train_l2, train_l3

def select_features():

    print ('Reading Data[1]...')
    train_date_part = pd.read_csv('input/train_date.csv', nrows = 10000)

    print ('')

    print ('Analyzing Date[1]...')
    date_cols = train_date_part.drop('Id', axis=1).count().reset_index().sort_values(by=0, ascending=False)
    print ('Analyzing Date[2]...')
    date_cols['station'] = date_cols['index'].apply(lambda s: s.split('_')[1])
    print (date_cols['station'])
    print ('Analyzing Date[3]...')
    date_cols = date_cols.drop_duplicates('station', keep='first')['index'].tolist()
    final_date = ['Id']

    print ('')

    for column in date_cols:
        if column not in ['Id']:
            print ('Date Column #:', column)
            list.append(final_date, column)

    print ('')

    print ('Reading Data[2]...')
    train_num = pd.read_csv('input/train_numeric.csv', nrows=100000, dtype=np.float32)
    print ('Reading Data[3]...')
    train_date = pd.read_csv('input/train_date.csv', nrows=100000, usecols=final_date, dtype=np.float32)
    print ('Reading Data[4]...')
    train_x = pd.merge(train_num, train_date, on='Id', how='left')
    print ('Reading Data[5]...')
    train_y = train_num['Response'].values

    print ('')
    print (train_x.columns)
    print ('')

    drop_list = ['Id', 'Response']
    for column in train_x:
        if column in drop_list:
            train_x.drop(column, inplace=True, axis=1)

    print ('Fitting Classifier[1]...')
    clf = XGBClassifier(base_score=0.005)
    clf.fit(train_x, train_y)

    column_importance = clf.feature_importances_
    columns = train_x.columns
    column_dict = dict()
    column_list = []

    for row, value in zip(columns, column_importance):
        column_dict[row] = value

    print ('')

    column_dict = sorted(column_dict.items(), key=operator.itemgetter(1))
    for row, value in column_dict:
        print ('Feature:', row,'| Importance:', value)
        if value > 0.0:
            list.append(column_list, row)

    print ('')
    print ('Number of Features:', len(column_list))
    print ('')

    num_features = []
    date_features = []

    for column in column_list:
        if '_F' in column:
            list.append(num_features, column)
    for column in column_list:
        if '_D' in column:
            list.append(date_features, column)
    for value in ['Id', 'Response']:
        list.append(num_features, value)
    list.append(date_features, 'Id')

    return num_features, date_features

def xgb_model(num_features, date_features):

    print ('Reading Data[1]...')
    train_num = pd.read_csv('input/train_numeric.csv', usecols=num_features, dtype=np.float32)
    
    print ('')
    
    print ('Creating Features[Num_Stations_Used]')
    train_num['num_stations_used'] = train_num.apply(stations_used, axis=1)
    print ('Creating Features[Num_Stations_Unused]')
    train_num['num_stations_unused'] = train_num.apply(stations_unused, axis=1)
    print ('Creating Features[Ratio_Used]')
    train_num['ratio_used'] = train_num['num_stations_used'] / (train_num['num_stations_used'] + train_num['num_stations_unused'])
    print ('Creating Features[Ratio_Unused]')
    train_num['ratio_unused'] = train_num['num_stations_unused'] / (train_num['num_stations_used'] + train_num['num_stations_unused'])

    train_l0, train_l1, train_l2, train_l3 = seperate_levels(train_num)
    print ('Creating Features[L0_Stations_Used]')
    train_num['L0_stations_used'] = train_l0.apply(stations_used, axis=1)
    print ('Creating Features[L1_Stations_Used]')
    train_num['L1_stations_used'] = train_l1.apply(stations_used, axis=1)
    print ('Creating Features[L2_Stations_Used]')
    train_num['L2_stations_used'] = train_l2.apply(stations_used, axis=1)
    print ('Creating Features[L3_Stations_Used]')
    train_num['L3_stations_used'] = train_l3.apply(stations_used, axis=1)

    print ('')
    print ('Creating Features[Path]')
    numeric = pd.read_csv('input/train_numeric.csv', usecols=num_features, dtype=np.float32)

    path_dict = dict()
    value_list = []
    path_number = 1

    for index, row in numeric.iterrows():

        idx = 0
        column_list = []

        for column in row:

            if np.isnan(column) == True:
                continue
            else:
                list.append(column_list, row.index[idx])
            idx += 1

        if tuple(column_list) in path_dict:
            pass
        else:
            path_dict[tuple(column_list)] = path_number
            path_number += 1

        list.append(value_list, path_dict[tuple(column_list)])

    numeric['path'] = value_list
    train_num['path'] = value_list

    print ('Creating Features[Path_Error]')
    print ('')

    error_list = []
    unique_path = dict()
    for path in numeric['path']:
        if path in unique_path:
            list.append(error_list, unique_path[path])
        else:
            train_path = numeric[numeric['path'] == path]
            error_rate = train_path[train_path.Response == 1].size / float(train_path[train_path.Response == 0].size)
            unique_path[path] = error_rate
            list.append(error_list, error_rate)

    numeric['path_error'] = error_list
    train_num['path_error'] = error_list

    print ('Reading Data[2]...')
    train_date = pd.read_csv('input/train_date.csv', usecols=date_features, dtype=np.float32)
    print ('Reading Data[3]...')
    train_x = pd.merge(train_num, train_date, on='Id', how='outer')
    print ('Reading Data[4]...')
    train_y = train_num['Response'].values

    num_features.remove('Response')

    print ('Reading Data[1]...')
    test_num = pd.read_csv('input/test_numeric.csv', usecols=num_features, dtype=np.float32)

    print ('')

    print ('Creating Features[Num_Stations_Used]')
    test_num['num_stations_used'] = test_num.apply(stations_used, axis=1)
    print ('Creating Features[Num_Stations_Unused]')
    test_num['num_stations_unused'] = test_num.apply(stations_unused, axis=1)
    print ('Creating Features[Ratio_Used]')
    test_num['ratio_used'] = test_num['num_stations_used'] / (test_num['num_stations_used'] + test_num['num_stations_unused'])
    print ('Creating Features[Ratio_Unused]')
    test_num['ratio_unused'] = test_num['num_stations_unused'] / (test_num['num_stations_used'] + test_num['num_stations_unused'])

    test_l0, test_l1, test_l2, test_l3 = seperate_levels(test_num)
    print ('Creating Features[L0_Stations_Used]')
    test_num['L0_stations_used'] = test_l0.apply(stations_used, axis=1)
    print ('Creating Features[L1_Stations_Used]')
    test_num['L1_stations_used'] = test_l1.apply(stations_used, axis=1)
    print ('Creating Features[L2_Stations_Used]')
    test_num['L2_stations_used'] = test_l2.apply(stations_used, axis=1)
    print ('Creating Features[L3_Stations_Used]')
    test_num['L3_stations_used'] = test_l3.apply(stations_used, axis=1)

    print ('')
    print ('Creating Features[Path]')
    numeric = pd.read_csv('input/test_numeric.csv', usecols=num_features, dtype=np.float32)

    value_list = []

    for index, row in numeric.iterrows():

        idx = 0
        column_list = []

        for column in row:

            if np.isnan(column) == True:
                continue
            else:
                list.append(column_list, row.index[idx])
            idx += 1

        if tuple(column_list) in path_dict:
            pass
        else:
            path_dict[tuple(column_list)] = path_number
            path_number += 1

        list.append(value_list, path_dict[tuple(column_list)])

    numeric['path'] = value_list
    test_num['path'] = value_list

    print ('Creating Features[Path_Error]')
    print ('')

    error_list = []
    for path in numeric['path']:
        if path in unique_path:
            list.append(error_list, unique_path[path])
        else:
            unique_path[path] = ''
            list.append(error_list, '')

    numeric['path_error'] = error_list
    test_num['path_error'] = error_list

    print ('Reading Data[2]...')
    test_date = pd.read_csv('input/test_date.csv', usecols=date_features, dtype=np.float32)
    print ('Reading Data[3]...')
    test_x = pd.merge(test_num, test_date, on='Id', how='outer')

    drop_list = ['Id', 'Response']
    for column in train_x:
        if column in drop_list:
            train_x.drop(column, inplace=True, axis=1)
    for column in test_x:
        if column in drop_list:
            test_x.drop(column, inplace=True, axis=1)

    clf = XGBClassifier(max_depth=5, base_score=.005)

    cv = StratifiedKFold(train_y, n_folds=3)
    preds = np.ones(train_y.shape[0])

    for i, (train, test) in enumerate(cv):

        print ('Fitting Classifier...')
        preds[test] = clf.fit(np.array(train_x)[train], train_y[train]).predict_proba(np.array(train_x)[test])[:,1]
        importance = clf.feature_importances_
        print("fold {}, ROC AUC: {:.3f}".format(i, roc_auc_score(train_y[test], preds[test])))

        print ('')
        columns = train_x.columns
        column_dict = dict()

        for column, importance in zip(columns, importance):
            column_dict[column] = importance
        sorted_dict = sorted(column_dict.items(), key=operator.itemgetter(1))
        for key, value in sorted_dict:
            print ("Column: ", key, "| Importance: ", value)
        print ('')

    print (roc_auc_score(train_y, preds))
    thresholds = np.linspace(0.01, 0.99, 50)
    mcc = np.array([matthews_corrcoef(train_y, preds>thr) for thr in thresholds])
    best_threshold = thresholds[mcc.argmax()]
    print (mcc.max())

    preds = (clf.predict_proba(np.array(test_x))[:,1] > best_threshold).astype(np.int8)
    sub = pd.read_csv('input/sample_submission.csv')
    output = pd.DataFrame()
    output['Id'] = sub['Id']
    output["Response"] = preds
    output.to_csv('output/xgb_model.csv', index=False)

if __name__ == '__main__':

    num_features, date_features = select_features()
    xgb_model(num_features, date_features)