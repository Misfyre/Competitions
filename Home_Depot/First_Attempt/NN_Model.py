import time
start_time = time.time()

import pandas as pd
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.constraints import nonneg
from keras.layers.advanced_activations import LeakyReLU

#import theano
#theano.config.openmp = False

print ('')

def rmse(y_true, y_pred):
    from keras import backend as k
    from keras.objectives import mean_squared_error

    return k.sqrt(mean_squared_error(y_true, y_pred))
def data_gen(df_all, y_train=None, n_batch=5, loop_forever=True):
    res = []
    y_res = []
    while True:
        pos = 0
        for row in df_all.itertuples():
            search_term = str(row.search_term)
            bag = str(row.bag)

            arr = np.ndarray((2, 64, 4800), np.float32)
            arr[0].fill(128)
            arr[1].fill(255)
            for i in range(len(search_term)):
                letter_i = search_term[i]
                val_i = ord(letter_i)

                for j in range(len(bag)):
                    letter_j = bag[j]
                    val_j = ord(letter_j)

                    arr[0, i, j] = val_i - val_j
                    arr[1, i, j] = val_i + val_j

            arr = arr.reshape((2, 640, 480))
            res.append(arr)
            if y_train is not None:
                y_res.append(y_train[pos])
                pos += 1

            if len(res) == n_batch:
                res = np.asarray(res)
                if y_train is not None:
                    y_res = np.asarray(y_res)
                    yield (res, y_res)
                    y_res = []
                else:
                    yield (res)

                res = []

        if not loop_forever:
            if len(res) > 0:
                res = np.asarray(res)
                if y_train is not None:
                    y_res = np.asarray(y_res)
                    yield (res, y_res)
                    y_res = []
                else:
                    yield (res)

            break
def batch_test(model, data_x, data_y, n_batch=5):
    foo = data_gen(data_x, data_y, n_batch=n_batch, loop_forever=False)
    res = []
    for x, y in foo:
        ans = model.test_on_batch(x, y)
        res.extend(ans)
        print('{:5.4f} - {:5.4f}'.format(np.mean(res), np.std(res)))
    return res
def batch_predict(model, data, n_batch=10):
    foo = data_gen(data, n_batch=n_batch, loop_forever=False)
    total = data.shape[0]
    res = []
    for x in foo:
        ans = model.predict(x).ravel().tolist()
        res.extend(ans)
        print('{:d} de {:d}'.format(len(res), total))
    return res

df_all = pd.read_csv(os.path.expanduser('~/Desktop/Home Depot/Features/feature_array.csv'))
feature_cols = [col for col in df_all.columns if col not in ['Unnamed: 0']]
df_all = df_all[feature_cols]

df_attr = pd.read_csv('~/Desktop/Home Depot/Data/attributes.csv')
df_attr.dropna(inplace=True)

material = dict()
df_attr['about_material'] = df_attr['name'].str.lower().str.contains('material')
for row in df_attr[df_attr['about_material']].iterrows():
    r = row[1]
    product = r['product_uid']
    value = r['value']
    material.setdefault(product, '')
    material[product] = material[product] + '' + str(value)
df_material = pd.DataFrame.from_dict(material, orient='index')
df_material = df_material.reset_index()
df_material.columns = ['product_uid', 'material']

color = dict()
df_attr['about_color'] = df_attr['name'].str.lower().str.contains('color')
for row in df_attr[df_attr['about_color']].iterrows():
    r = row[1]
    product = r['product_uid']
    value = r['value']
    color.setdefault(product, '')
    color[product] = color[product] + '' + str(value)
df_color = pd.DataFrame.from_dict(color, orient='index')
df_color = df_color.reset_index()
df_color.columns = ['product_uid', 'color']

df_all = pd.merge(df_all, df_material, how='left', on='product_uid')
df_all = pd.merge(df_all, df_color, how='left', on='product_uid')

print("--- Processing columns: %s minutes ---" % round(((time.time() - start_time)/60),2))

del df_color, df_material

df_all['search_term'] = df_all['search_term']
df_all['bag'] = df_all['product_title'].str.pad(df_all['product_title'].str.len().max(), side='right') + ' ' + \
                df_all['product_description'].str.pad(df_all['product_description'].str.len().max(), side='right') + ' ' + \
                df_all['brand'].str.pad(df_all['brand'].str.len().max(), side='right') + ' ' + \
                df_all['material'].str.pad(df_all['material'].str.len().max(), side='right') + ' ' + \
                df_all['color'].str.pad(df_all['color'].str.len().max(), side='right')

print("--- Stemming columns: %s minutes ---" % round(((time.time() - start_time)/60),2))

df_train = pd.read_csv('~/Desktop/Home Depot/Data/train.csv', encoding="ISO-8859-1")[:1000]
num_train = df_train.shape[0]

df_train = df_all[:num_train]
df_test = df_all[num_train:]

model = Sequential()
model.add(BatchNormalization(input_shape=(2, 640, 480)))
model.add(Convolution2D(16, 3, 3, border_mode='valid'))
model.add(LeakyReLU())
model.add(Convolution2D(16, 3, 3))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(LeakyReLU())

model.add(Dense(1, W_constraint=nonneg()))
model.add(Activation('linear'))

model.compile(loss=rmse, optimizer='adam')

print("--- Compiling model: %s minutes ---" % round(((time.time() - start_time)/60),2))
print ('')

model.fit_generator(data_gen(df_train, y_train, n_batch=5),
                    samples_per_epoch=df_train.shape[0],
                    nb_epoch=1,
                    nb_worker=4,
                    )

print("--- Fitting model: %s minutes ---" % round(((time.time() - start_time)/60),2))

id_test = df_test['id']
y_pred = batch_predict(model, df_test, n_batch=20)
y_pred = np.asarray(y_pred)
print(y_pred)
