import warnings
warnings.filterwarnings("ignore")

print ('')
print ("Initializing Libraries...")

import pandas as pd
import numpy as np
import sys
import os

from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

from keras.layers.advanced_activations import PReLU
from keras.models import Sequential
from keras.layers import Dense, Dropout

#--------------------------------------------------------Models--------------------------------------------------------#

def model1(X_train):

    model = Sequential()
    model.add(Dense(150, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(50, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(12, init='normal', activation='softmax'))

    return model

def model2(X_train):

    model = Sequential()
    model.add(Dense(200, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.5))
    model.add(Dense(80, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.3))
    model.add(Dense(12, init='normal', activation='softmax'))

    return model

def model3(X_train):

    model = Sequential()
    model.add(Dense(180, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(50, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.1))
    model.add(Dense(12, init='normal', activation='softmax'))

    return model

def model4(X_train):

    model = Sequential()
    model.add(Dense(250, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.3))
    model.add(Dense(80, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(12, init='normal', activation='softmax'))

    return model

def model5(X_train):

    model = Sequential()
    model.add(Dense(120, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(80, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.3))
    model.add(Dense(12, init='normal', activation='softmax'))

    return model

def model6(X_train):

    model = Sequential()
    model.add(Dense(150, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.6))
    model.add(Dense(100, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(12, init='normal', activation='softmax'))

    return model

def model7(X_train):

    model = Sequential()
    model.add(Dense(175, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.42))
    model.add(Dense(90, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.36))
    model.add(Dense(12, init='normal', activation='softmax'))

    return model

def model8(X_train):

    model = Sequential()
    model.add(Dense(205, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.62))
    model.add(Dense(55, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.29))
    model.add(Dense(12, init='normal', activation='softmax'))

    return model

def model9(X_train):

    model = Sequential()
    model.add(Dense(250, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.7))
    model.add(Dense(150, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(12, init='normal', activation='softmax'))

    return model

def model10(X_train):

    model = Sequential()
    model.add(Dense(165, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.35))
    model.add(Dense(120, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.21))
    model.add(Dense(12, init='normal', activation='softmax'))

    return model

def model11(X_train):

    model = Sequential()
    model.add(Dense(180, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.62))
    model.add(Dense(65, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.24))
    model.add(Dense(12, init='normal', activation='softmax'))

    return model

def model12(X_train):

    model = Sequential()
    model.add(Dense(120, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.66))
    model.add(Dense(45, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(12, init='normal', activation='softmax'))

    return model

def model13(X_train):

    model = Sequential()
    model.add(Dense(300, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.8))
    model.add(Dense(150, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.6))
    model.add(Dense(12, init='normal', activation='softmax'))

    return model

def model14(X_train):

    model = Sequential()
    model.add(Dense(275, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(150, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.1))
    model.add(Dense(12, init='normal', activation='softmax'))

    return model

def model15(X_train):

    model = Sequential()
    model.add(Dense(240, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.6))
    model.add(Dense(145, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.34))
    model.add(Dense(12, init='normal', activation='softmax'))

    return model

def model16(X_train):

    model = Sequential()
    model.add(Dense(350, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.74))
    model.add(Dense(180, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.42))
    model.add(Dense(12, init='normal', activation='softmax'))

    return model

def model17(X_train):

    model = Sequential()
    model.add(Dense(235, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.43))
    model.add(Dense(145, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.26))
    model.add(Dense(12, init='normal', activation='softmax'))

    return model

def model18(X_train):

    model = Sequential()
    model.add(Dense(150, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.42))
    model.add(Dense(30, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.19))
    model.add(Dense(12, init='normal', activation='softmax'))

    return model

def model19(X_train):

    model = Sequential()
    model.add(Dense(275, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.8))
    model.add(Dense(150, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.5))
    model.add(Dense(12, init='normal', activation='softmax'))

    return model

def model20(X_train):

    model = Sequential()
    model.add(Dense(215, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.76))
    model.add(Dense(140, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.43))
    model.add(Dense(12, init='normal', activation='softmax'))

    return model

#-------------------------------------------------------Analysis-------------------------------------------------------#

def rstr(df):

    return df.dtypes, df.head(3) ,df.apply(lambda x: [x.unique()]), df.apply(lambda x: [len(x.unique())]),df.shape

def batch_generator(X, y, batch_size, shuffle):

    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):

    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

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

    print ("#Expanding to Multiple Rows")
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

#------------------------------------------------------Initialize------------------------------------------------------#

def initialize_model(model, nb_epoch, X_train, X_val, y_train, y_val, test_sp, label_group, device_id):

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    fit = model.fit_generator(generator=batch_generator(X_train, y_train, 400, True), nb_epoch=nb_epoch, samples_per_epoch=70496, validation_data=(X_val.todense(), y_val), verbose=2)
    scores_val = model.predict_generator(generator=batch_generatorp(X_val, 400, False), val_samples=X_val.shape[0])
    print ('logloss val {}'.format(log_loss(y_val, scores_val)))

    print ("# Final prediction")
    scores = model.predict_generator(generator=batch_generatorp(test_sp, 800, False), val_samples=test_sp.shape[0])
    result = pd.DataFrame(scores , columns=label_group.classes_)
    result["device_id"] = device_id
    result = result.set_index("device_id")

    return result

if __name__ == '__main__':

    X_train, X_val, y_train, y_val, test_sp, label_group, device_id = convert_data()

    model1 = model1(X_train)
    model2 = model2(X_train)
    model3 = model3(X_train)
    model4 = model4(X_train)
    model5 = model5(X_train)
    model6 = model6(X_train)
    model7 = model7(X_train)
    model8 = model8(X_train)
    model9 = model9(X_train)
    model10 = model10(X_train)

    model11 = model11(X_train)
    model12 = model12(X_train)
    model13 = model13(X_train)
    model14 = model14(X_train)
    model15 = model15(X_train)
    model16 = model16(X_train)
    model17 = model17(X_train)
    model18 = model18(X_train)
    model19 = model19(X_train)
    model20 = model20(X_train)

    result1 = initialize_model(model1, 16, X_train, X_val, y_train, y_val, test_sp, label_group, device_id)
    result2 = initialize_model(model2, 14, X_train, X_val, y_train, y_val, test_sp, label_group, device_id)
    result3 = initialize_model(model3, 18, X_train, X_val, y_train, y_val, test_sp, label_group, device_id)
    result4 = initialize_model(model4, 16, X_train, X_val, y_train, y_val, test_sp, label_group, device_id)
    result5 = initialize_model(model5, 13, X_train, X_val, y_train, y_val, test_sp, label_group, device_id)
    result6 = initialize_model(model6, 17, X_train, X_val, y_train, y_val, test_sp, label_group, device_id)
    result7 = initialize_model(model7, 14, X_train, X_val, y_train, y_val, test_sp, label_group, device_id)
    result8 = initialize_model(model8, 14, X_train, X_val, y_train, y_val, test_sp, label_group, device_id)
    result9 = initialize_model(model9, 29, X_train, X_val, y_train, y_val, test_sp, label_group, device_id)
    result10 = initialize_model(model10, 10, X_train, X_val, y_train, y_val, test_sp, label_group, device_id)

    result11 = initialize_model(model11, 27, X_train, X_val, y_train, y_val, test_sp, label_group, device_id)
    result12 = initialize_model(model12, 31, X_train, X_val, y_train, y_val, test_sp, label_group, device_id)
    result13 = initialize_model(model13, 24, X_train, X_val, y_train, y_val, test_sp, label_group, device_id)
    result14 = initialize_model(model14, 11, X_train, X_val, y_train, y_val, test_sp, label_group, device_id)
    result15 = initialize_model(model15, 20, X_train, X_val, y_train, y_val, test_sp, label_group, device_id)
    result16 = initialize_model(model16, 26, X_train, X_val, y_train, y_val, test_sp, label_group, device_id)
    result17 = initialize_model(model17, 14, X_train, X_val, y_train, y_val, test_sp, label_group, device_id)
    result18 = initialize_model(model18, 18, X_train, X_val, y_train, y_val, test_sp, label_group, device_id)
    result19 = initialize_model(model19, 29, X_train, X_val, y_train, y_val, test_sp, label_group, device_id)
    result20 = initialize_model(model20, 25, X_train, X_val, y_train, y_val, test_sp, label_group, device_id)

    final_result = pd.DataFrame()

    final_result['F23-'] = (result1['F23-'] + result2['F23-'] + result3['F23-'] + result4['F23-'] + result5['F23-'] + result6['F23-'] + result7['F23-'] + result8['F23-'] + result9['F23-'] + result10['F23-'] + result11['F23-'] + result12['F23-'] + result13['F23-'] + result14['F23-'] + result15['F23-'] + result16['F23-'] + result17['F23-'] + result18['F23-'] + result19['F23-'] + result20['F23-']) / 20
    final_result['F24-26'] = (result1['F24-26'] + result2['F24-26'] + result3['F24-26'] + result4['F24-26'] + result5['F24-26'] + result6['F24-26'] + result7['F24-26'] + result8['F24-26'] + result9['F24-26'] + result10['F24-26'] + result11['F24-26'] + result12['F24-26'] + result13['F24-26'] + result14['F24-26'] + result15['F24-26'] + result16['F24-26'] + result17['F24-26'] + result18['F24-26'] + result19['F24-26'] + result20['F24-26']) / 20
    final_result['F27-28'] = (result1['F27-28'] + result2['F27-28'] + result3['F27-28'] + result4['F27-28'] + result5['F27-28'] + result6['F27-28'] + result7['F27-28'] + result8['F27-28'] + result9['F27-28'] + result10['F27-28'] + result11['F27-28'] + result12['F27-28'] + result13['F27-28'] + result14['F27-28'] + result15['F27-28'] + result16['F27-28'] + result17['F27-28'] + result18['F27-28'] + result19['F27-28'] + result20['F27-28']) / 20
    final_result['F29-32'] = (result1['F29-32'] + result2['F29-32'] + result3['F29-32'] + result4['F29-32'] + result5['F29-32'] + result6['F29-32'] + result7['F29-32'] + result8['F29-32'] + result9['F29-32'] + result10['F29-32'] + result11['F29-32'] + result12['F29-32'] + result13['F29-32'] + result14['F29-32'] + result15['F29-32'] + result16['F29-32'] + result17['F29-32'] + result18['F29-32'] + result19['F29-32'] + result20['F29-32']) / 20
    final_result['F33-42'] = (result1['F33-42'] + result2['F33-42'] + result3['F33-42'] + result4['F33-42'] + result5['F33-42'] + result6['F33-42'] + result7['F33-42'] + result8['F33-42'] + result9['F33-42'] + result10['F33-42'] + result11['F33-42'] + result12['F33-42'] + result13['F33-42'] + result14['F33-42'] + result15['F33-42'] + result16['F33-42'] + result17['F33-42'] + result18['F33-42'] + result19['F33-42'] + result20['F33-42']) / 20
    final_result['F43+'] = (result1['F43+'] + result2['F43+'] + result3['F43+'] + result4['F43+'] + result5['F43+'] + result6['F43+'] + result7['F43+'] + result8['F43+'] + result9['F43+'] + result10['F43+'] + result11['F43+'] + result12['F43+'] + result13['F43+'] + result14['F43+'] + result15['F43+'] + result16['F43+'] + result17['F43+'] + result18['F43+'] + result19['F43+'] + result20['F43+']) / 20
    final_result['M22-'] = (result1['M22-'] + result2['M22-'] + result3['M22-'] + result4['M22-'] + result5['M22-'] + result6['M22-'] + result7['M22-'] + result8['M22-'] + result9['M22-'] + result10['M22-'] + result11['M22-'] + result12['M22-'] + result13['M22-'] + result14['M22-'] + result15['M22-'] + result16['M22-'] + result17['M22-'] + result18['M22-'] + result19['M22-'] + result20['M22-']) / 20
    final_result['M23-26'] = (result1['M23-26'] + result2['M23-26'] + result3['M23-26'] + result4['M23-26'] + result5['M23-26'] + result6['M23-26'] + result7['M23-26'] + result8['M23-26'] + result9['M23-26'] + result10['M23-26'] + result11['M23-26'] + result12['M23-26'] + result13['M23-26'] + result14['M23-26'] + result15['M23-26'] + result16['M23-26'] + result17['M23-26'] + result18['M23-26'] + result19['M23-26'] + result20['M23-26']) / 20
    final_result['M27-28'] = (result1['M27-28'] + result2['M27-28'] + result3['M27-28'] + result4['M27-28'] + result5['M27-28'] + result6['M27-28'] + result7['M27-28'] + result8['M27-28'] + result9['M27-28'] + result10['M27-28'] + result11['M27-28'] + result12['M27-28'] + result13['M27-28'] + result14['M27-28'] + result15['M27-28'] + result16['M27-28'] + result17['M27-28'] + result18['M27-28'] + result19['M27-28'] + result20['M27-28']) / 20
    final_result['M29-31'] = (result1['M29-31'] + result2['M29-31'] + result3['M29-31'] + result4['M29-31'] + result5['M29-31'] + result6['M29-31'] + result7['M29-31'] + result8['M29-31'] + result9['M29-31'] + result10['M29-31'] + result11['M29-31'] + result12['M29-31'] + result13['M29-31'] + result14['M29-31'] + result15['M29-31'] + result16['M29-31'] + result17['M29-31'] + result18['M29-31'] + result19['M29-31'] + result20['M29-31']) / 20
    final_result['M32-38'] = (result1['M32-38'] + result2['M32-38'] + result3['M32-38'] + result4['M32-38'] + result5['M32-38'] + result6['M32-38'] + result7['M32-38'] + result8['M32-38'] + result9['M32-38'] + result10['M32-38'] + result11['M32-38'] + result12['M32-38'] + result13['M32-38'] + result14['M32-38'] + result15['M32-38'] + result16['M32-38'] + result17['M32-38'] + result18['M32-38'] + result19['M32-38'] + result20['M32-38']) / 20
    final_result['M39+'] = (result1['M39+'] + result2['M39+'] + result3['M39+'] + result4['M39+'] + result5['M39+'] + result6['M39+'] + result7['M39+'] + result8['M39+'] + result9['M39+'] + result10['M39+'] + result11['M39+'] + result12['M39+'] + result13['M39+'] + result14['M39+'] + result15['M39+'] + result16['M39+'] + result17['M39+'] + result18['M39+'] + result19['M39+'] + result20['M39+']) / 20

    final_result['device_id'] = device_id
    final_result = final_result.set_index('device_id')
    final_result.to_csv(os.path.expanduser('~/Desktop/TalkingData/Submissions/nn_ensemble.csv'))