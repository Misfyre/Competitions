import time
start_time = time.time()

import numpy as np
import pandas as pd
import re
import os

import random
random.seed(2016)

def seg_words(str1, str2):
    str2 = str2.lower()
    str2 = re.sub("[^a-z0-9./]"," ", str2)
    str2 = [z for z in set(str2.split()) if len(z)>2]
    words = str1.lower().split(" ")
    s = []
    for word in words:
        if len(word)>3:
            s1 = []
            s1 += segmentit(word,str2,True)
            if len(s)>1:
                s += [z for z in s1 if z not in ['er','ing','s','less'] and len(z)>1]
            else:
                s.append(word)
        else:
            s.append(word)
    return (" ".join(s))
def segmentit(s, txt_arr, t):
    st = s
    r = []
    for j in range(len(s)):
        for word in txt_arr:
            if word == s[:-j]:
                r.append(s[:-j])
                s=s[len(s)-j:]
                r += segmentit(s, txt_arr, False)
    if t:
        i = len(("").join(r))
        if not i==len(st):
            r.append(st[i:])
    return r
def str_common_word(str1, str2):
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word)>=0:
            cnt+=1
    return cnt
def str_whole_word(str1, str2, i_):
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt
def str_entire_word(str1, str2):
    cnt = 0
    if str1 in str2:
        cnt = 1
    else:
        cnt = 0
    return cnt

df_all = pd.read_csv(os.path.expanduser('~/Desktop/Home Depot/Features/initial_features.csv'))#[:2000]

df_all['product_info'] = df_all['search_term']+"\t"+ df_all['product_title'] +"\t"+ df_all['product_description'] +"\t"+ df_all['brand']

df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_title'] = df_all['product_title'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_description'] = df_all['product_description'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_brand'] = df_all['brand'].map(lambda x:len(x.split())).astype(np.int64)

df_all['search_term'] = df_all['product_info'].map(lambda x:seg_words(x.split('\t')[0],x.split('\t')[1]))
df_all['query_in_title'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[1],0))
df_all['query_in_description'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[2],0))
df_all['query_in_brand'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[3],0))

df_all['query_first_word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[0],x.split('\t')[1]))
df_all['query_first_word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[0],x.split('\t')[2]))
df_all['query_first_word_in_brand'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[0],x.split('\t')[3]))

df_all['query_last_word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[1]))
df_all['query_last_word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[2]))
df_all['query_last_word_in_brand'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[3]))

df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
df_all['word_in_brand'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[3]))

df_all['ratio_title'] = df_all['word_in_title']/df_all['len_of_query']
df_all['ratio_description'] = df_all['word_in_description']/df_all['len_of_query']
df_all['ratio_brand'] = df_all['word_in_brand']/df_all['len_of_query']

df_all['title_len_ratio'] = df_all['word_in_title']/df_all['len_of_title']
df_all['description_len_ratio'] = df_all['word_in_description']/df_all['len_of_description']
df_all['brand_len_ratio'] = df_all['word_in_brand']/df_all['len_of_brand']

df_all['attr'] = df_all['search_term']+"\t"+df_all['brand']
df_all['query_occurences_in_title'] = df_all['product_info'].map(lambda x:str_entire_word(x.split('\t')[0],x.split('\t')[1]))

df_brand = pd.unique(df_all.brand.ravel())
d={}
i = 1000
for s in df_brand:
    d[s]=i
    i+=3

df_all['brand_feature'] = df_all['brand'].map(lambda x:d[x])
df_all['search_term_feature'] = df_all['search_term'].map(lambda x:len(x))

feature_cols_all = [col for col in df_all.columns if col not in ['Unnamed: 0']]
df_all[feature_cols_all].to_csv(os.path.expanduser('~/Desktop/Home Depot/Features/feature_array.csv'))

print ('')
print ("--- Features Set: %s minutes ---" % round(((time.time() - start_time)/60),2))
