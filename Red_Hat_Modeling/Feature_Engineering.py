__author__ = "Nick Sarris"

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
import gc

print('')

def process_activity_category(input_df):
    df = input_df.copy()
    return df.assign(activity_category=lambda df:
                     df.activity_category.str.lstrip('type ').astype(np.int32))

def process_activities_char(input_df, columns_range):
    df = input_df.copy()
    char_columns = ['char_' + str(i) for i in columns_range]
    return (df[char_columns].fillna('type -999')
            .apply(lambda col: col.str.lstrip('type ').astype(np.int32))
            .join(df.drop(char_columns, axis=1)))

def process_people_bool_char(input_df, columns_range):
    df = input_df.copy()
    boolean_char_columns = ['char_' + str(i) for i in columns_range]
    return (df[boolean_char_columns].apply(lambda col: col.astype(np.int32))
                                    .join(df.drop(boolean_char_columns,
                                                  axis=1)))

def process_people_cat_char(input_df, columns_range):
    df = input_df.copy()
    cat_char_columns = ['char_' + str(i) for i in columns_range]
    return (df[cat_char_columns].apply(lambda col:
                                       col.str.lstrip('type ').astype(np.int32))
                                .join(df.drop(cat_char_columns, axis=1)))

def process_group_1(input_df):
    df = input_df.copy()
    return df.assign(group_1=lambda df:
                     df.group_1.str.lstrip('group ').astype(np.int32))

def people_processing(input_df):
    df = input_df.copy()
    return (df.pipe(process_group_1)
              .pipe(process_people_cat_char, range(1, 10))
              .pipe(process_people_bool_char, range(10, 38)))

def activities_processing(input_df):
    df = input_df.copy()
    return (df.pipe(process_activity_category)
              .pipe(process_activities_char, range(1, 11)))

print ('Initializing Step_1/10...')
train = pd.read_csv(('input/act_train.csv'), header = 0)
test = pd.read_csv(('input/act_test.csv'), header = 0)
people = pd.read_csv(('input/people.csv'), header = 0)
train_path = 'input/processed_train.csv'
test_path = 'input/processed_test.csv'

train = train.pipe(activities_processing)
test = test.pipe(activities_processing)
people = people.pipe(people_processing)

print ('Initializing Step_2/10...')
for x in [col for col in people.columns if people[col].dtype == np.dtype(bool)]:
     people[x] = people[x]*1

for k in range(1,10,1):
    people['char_{}'.format(k)]= pd.factorize(people['char_{}'.format(k)])[0]

print ('Initializing Step_3/10...')
train['day_of_week']=train.loc[:,'date'].apply(lambda x : datetime.strptime(str(x) ,'%Y-%m-%d').weekday())
train['month']=train.loc[:,'date'].apply(lambda x : datetime.strptime(str(x) ,'%Y-%m-%d').strftime('%B'))
train['year']=train.loc[:,'date'].apply(lambda x : datetime.strptime(str(x) ,'%Y-%m-%d').strftime('%Y'))

train.date = train.date.apply(lambda x : datetime.strptime(x , '%Y-%m-%d'))
people.date=people.date.apply(lambda x : datetime.strptime(x ,'%Y-%m-%d'))

print ('Initializing Step_4/10...')
train = pd.concat([train,pd.get_dummies(train.activity_category)] , axis=1)
train = pd.concat([train,pd.get_dummies(train.month)] , axis=1)
train = pd.concat([train,pd.get_dummies(train.year)] , axis=1)
train = pd.concat([train,pd.get_dummies(train.day_of_week)] , axis=1)

train.drop('month',axis=1,inplace=True)
train.drop('year',axis=1,inplace=True)
train.drop('day_of_week',axis=1,inplace=True)
train_data=pd.merge(train,people,on='people_id')

print ('Initializing Step_5/10...')
group = pd.DataFrame(train_data.groupby(['people_id','date_x' ,'activity_category']).size())
group.columns=['count_activity']

people_2=[]
people_3=[]
people_4=[]

for pep, df in group.groupby(level = 0):

    if 2 in df.count_activity.values:
        people_2.append(pep)

    if 3 in df.count_activity.values:
        people_3.append(pep)

    if 4 in df.count_activity.values:
        people_4.append(pep)

del group

print ('Initializing Step_6/10...')
t_1 = set(people_2)
train_data['t_2_activities'] = train_data.people_id.apply(lambda x : set([x]).intersection(t_1)==set([x]))
t_2 = set(people_3)
train_data['t_3_activities'] = train_data.people_id.apply(lambda x : set([x]).intersection(t_2)==set([x]))
t_3 = set(people_4)
train_data['t_4_activities'] = train_data.people_id.apply(lambda x : set([x]).intersection(t_3)==set([x]))

group = pd.DataFrame(train_data.groupby(['people_id','activity_category']).size())
group.columns=['act_count']

same_activ_2 =[]
same_activ_4 =[]
same_activ_6 =[]
same_activ_8 =[]
same_activ_10 =[]

for pep,df in group.groupby(level=0):

    if any(df.act_count.values >9):
        same_activ_10.append(pep)

    elif any(df.act_count.values >7):
        same_activ_8.append(pep)

    elif any(df.act_count.values >5):
        same_activ_6.append(pep)

    elif any(df.act_count.values >3):
        same_activ_4.append(pep)

    elif any(df.act_count.values >1):
        same_activ_2.append(pep)

    else:
        pass

del group

print ('Initializing Step_7/10...')
s_1 = set(same_activ_2)
train_data['same_activity_2'] = train_data.people_id.apply(lambda x : set([x]).intersection(s_1)==set([x]))
s_2 = set(same_activ_4)
train_data['same_activity_4'] = train_data.people_id.apply(lambda x : set([x]).intersection(s_2)==set([x]))
s_3 = set(same_activ_6)
train_data['same_activity_6'] = train_data.people_id.apply(lambda x : set([x]).intersection(s_3)==set([x]))
s_4 = set(same_activ_8)
train_data['same_activity_8'] = train_data.people_id.apply(lambda x : set([x]).intersection(s_4)==set([x]))
s_5 = set(same_activ_10)
train_data['same_activity_10'] = train_data.people_id.apply(lambda x : set([x]).intersection(s_5)==set([x]))

activities_2=[]
activities_4=[]
activities_6=[]
activities_8=[]
activities_10=[]

tet = pd.DataFrame(train_data.groupby(['people_id','date_x'])['activity_category'].agg({'counts_the_activities':np.size}))
for pep , df in tet.groupby(level=0):

    if 2 & 3 in df.counts_the_activities.values:
        activities_2.append(pep)

    if 4 & 5 in df.counts_the_activities.values:
        activities_4.append(pep)

    if 6 & 7 in df.counts_the_activities.values:
        activities_6.append(pep)

    if 8 & 9 in df.counts_the_activities.values:
        activities_8.append(pep)

    if any(df.counts_the_activities.values>9):
        activities_10.append(pep)

del tet

print ('Initializing Step_8/10...')
r_1 = set(activities_2)
train_data['same_time_activ_2'] = train_data.people_id.apply(lambda x : set([x]).intersection(r_1)==set([x]))
r_2 = set(activities_4)
train_data['same_time_activ_4'] = train_data.people_id.apply(lambda x : set([x]).intersection(r_2)==set([x]))
r_3 = set(activities_6)
train_data['same_time_activ_6'] = train_data.people_id.apply(lambda x : set([x]).intersection(r_3)==set([x]))
r_4 = set(activities_8)
train_data['same_time_activ_8'] = train_data.people_id.apply(lambda x : set([x]).intersection(r_4)==set([x]))
r_5 = set(activities_10)
train_data['same_time_activ_10'] = train_data.people_id.apply(lambda x : set([x]).intersection(r_5)==set([x]))

train_data['occur']= train_data.people_id
train_data.occur = train_data.people_id.apply(dict(train_data.people_id.value_counts()).get)

print ('Initializing Step_9/10...')
train_data=pd.merge(train_data,people.loc[:,['people_id','mean_time']],on='people_id')

first_activitie = train_data.loc[:,['people_id','date_x','activity_category']].sort(columns=['people_id','date_x']).drop_duplicates(['people_id'] ,keep='first')
first_activitie.rename(columns = {'activity_category':'first activity'} , inplace = True)
first_activitie.drop('date_x',axis=1,inplace=True)

last_activity = train_data.loc[:,['people_id','date_x','activity_category']].sort(columns=['people_id','date_x']).drop_duplicates(['people_id'],keep='last')
last_activity.rename(columns = {'activity_category':'last_activity'} , inplace=True)
last_activity.drop('date_x',axis=1,inplace=True)

train_data = pd.merge(train_data,first_activitie,on='people_id')
train_data = pd.merge(train_data,last_activity,on='people_id')

del last_activity, first_activitie
gc.collect()

print ('Initializing Step_10/10...')
people_group =train_data.groupby('people_id')

frame_x=pd.DataFrame(people_group['date_x'].agg({'min_date_x':np.min}))
frame_y=pd.DataFrame(people_group['date_y'].agg({'min_date_y':np.min}))
frame_x.reset_index(level='people_id',inplace=True)
frame_y.reset_index(level='people_id',inplace=True)

frame=pd.merge(frame_x,frame_y,on='people_id')
frame['time_diff']=((frame.min_date_x -frame.min_date_y)/np.timedelta64(1,'D')).astype(int)
train_data=pd.merge(train_data,frame.loc[:,['people_id','time_diff']],on='people_id')

del people_group, frame

print ('')

people = pd.read_csv(('input/people.csv'), header = 0)
people = people.pipe(people_processing)

print ('Initializing Step_1/9...')
for x in [col for col in people.columns if people[col].dtype == np.dtype(bool)]:
     people[x] = people[x]*1

for k in range(1,10,1):
    people['char_{}'.format(k)]= pd.factorize(people['char_{}'.format(k)])[0]

print ('Initializing Step_2/9...')
test['day_of_week']=test.loc[:,'date'].apply(lambda x : datetime.strptime(str(x) ,'%Y-%m-%d').weekday())
test['month']=test.loc[:,'date'].apply(lambda x : datetime.strptime(str(x) ,'%Y-%m-%d').strftime('%B'))
test['year']=test.loc[:,'date'].apply(lambda x : datetime.strptime(str(x) ,'%Y-%m-%d').strftime('%Y'))

test.date = test.date.apply(lambda x : datetime.strptime(x , '%Y-%m-%d'))
people.date=people.date.apply(lambda x : datetime.strptime(x ,'%Y-%m-%d'))

print ('Initializing Step_3/9...')
test = pd.concat([test,pd.get_dummies(test.activity_category)] , axis=1)
test = pd.concat([test,pd.get_dummies(test.month)] , axis=1)
test = pd.concat([test,pd.get_dummies(test.year)] , axis=1)
test = pd.concat([test,pd.get_dummies(test.day_of_week)] , axis=1)

test.drop('month',axis=1,inplace=True)
test.drop('year',axis=1,inplace=True)
test.drop('day_of_week',axis=1,inplace=True)
test_data=pd.merge(test,people,on='people_id')

del test

print ('Initializing Step_4/9...')
group = pd.DataFrame(test_data.groupby(['people_id','date_x' ,'activity_category']).size())
group.columns=['count_activity']

people_2=[]
people_3=[]
people_4=[]

for pep, df in group.groupby(level = 0):

    if 2 in df.count_activity.values:
        people_2.append(pep)

    if 3 in df.count_activity.values:
        people_3.append(pep)

    if 4 in df.count_activity.values:
        people_4.append(pep)

del group

print ('Initializing Step_5/9...')
t_1 = set(people_2)
test_data['t_2_activities'] = test_data.people_id.apply(lambda x : set([x]).intersection(t_1)==set([x]))
t_2 = set(people_3)
test_data['t_3_activities'] = test_data.people_id.apply(lambda x : set([x]).intersection(t_2)==set([x]))
t_3 = set(people_4)
test_data['t_4_activities'] = test_data.people_id.apply(lambda x : set([x]).intersection(t_3)==set([x]))

group = pd.DataFrame(test_data.groupby(['people_id','activity_category']).size())
group.columns=['act_count']

same_activ_2 =[]
same_activ_4 =[]
same_activ_6 =[]
same_activ_8 =[]
same_activ_10 =[]

for pep,df in group.groupby(level=0):

    if any(df.act_count.values >9):
        same_activ_10.append(pep)

    elif any(df.act_count.values >7):
        same_activ_8.append(pep)

    elif any(df.act_count.values >5):
        same_activ_6.append(pep)

    elif any(df.act_count.values >3):
        same_activ_4.append(pep)

    elif any(df.act_count.values >1):
        same_activ_2.append(pep)

    else:
        pass

del group

print ('Initializing Step_6/9...')
s_1 = set(same_activ_2)
test_data['same_activity_2'] = test_data.people_id.apply(lambda x : set([x]).intersection(s_1)==set([x]))
s_2 = set(same_activ_4)
test_data['same_activity_4'] = test_data.people_id.apply(lambda x : set([x]).intersection(s_2)==set([x]))
s_3 = set(same_activ_6)
test_data['same_activity_6'] = test_data.people_id.apply(lambda x : set([x]).intersection(s_3)==set([x]))
s_4 = set(same_activ_8)
test_data['same_activity_8'] = test_data.people_id.apply(lambda x : set([x]).intersection(s_4)==set([x]))
s_5 = set(same_activ_10)
test_data['same_activity_10'] = test_data.people_id.apply(lambda x : set([x]).intersection(s_5)==set([x]))

activities_2=[]
activities_4=[]
activities_6=[]
activities_8=[]
activities_10=[]

tet = pd.DataFrame(test_data.groupby(['people_id','date_x'])['activity_category'].agg({'counts_the_activities':np.size}))
for pep , df in tet.groupby(level=0):

    if 2 & 3 in df.counts_the_activities.values:
        activities_2.append(pep)

    if 4 & 5 in df.counts_the_activities.values:
        activities_4.append(pep)

    if 6 & 7 in df.counts_the_activities.values:
        activities_6.append(pep)

    if 8 & 9 in df.counts_the_activities.values:
        activities_8.append(pep)

    if any(df.counts_the_activities.values>9):
        activities_10.append(pep)

del tet

print ('Initializing Step_7/9...')
r_1 = set(activities_2)
test_data['same_time_activ_2'] = test_data.people_id.apply(lambda x : set([x]).intersection(r_1)==set([x]))
r_2 = set(activities_4)
test_data['same_time_activ_4'] = test_data.people_id.apply(lambda x : set([x]).intersection(r_2)==set([x]))
r_3 = set(activities_6)
test_data['same_time_activ_6'] = test_data.people_id.apply(lambda x : set([x]).intersection(r_3)==set([x]))
r_4 = set(activities_8)
test_data['same_time_activ_8'] = test_data.people_id.apply(lambda x : set([x]).intersection(r_4)==set([x]))
r_5 = set(activities_10)
test_data['same_time_activ_10'] = test_data.people_id.apply(lambda x : set([x]).intersection(r_5)==set([x]))

test_data['occur']=test_data.people_id
test_data.occur=test_data.people_id.apply(dict(test_data.people_id.value_counts()).get)

print ('Initializing Step_8/9...')
test_data=pd.merge(test_data,people.loc[:,['people_id','mean_time']],on='people_id')

first_activitie = test_data.loc[:,['people_id','date_x','activity_category']].sort(columns=['people_id','date_x']).drop_duplicates(['people_id'] ,keep='first')
first_activitie.rename(columns = {'activity_category':'first activity'} , inplace = True)
first_activitie.drop('date_x',axis=1,inplace=True)

last_activity = test_data.loc[:,['people_id','date_x','activity_category']].sort(columns=['people_id','date_x']).drop_duplicates(['people_id'],keep='last')
last_activity.rename(columns = {'activity_category':'last_activity'} , inplace=True)
last_activity.drop('date_x',axis=1,inplace=True)

test_data = pd.merge(test_data,first_activitie,on='people_id')
test_data = pd.merge(test_data,last_activity,on='people_id')

del last_activity, first_activitie
gc.collect()

print ('Initializing Step_9/9...')
people_group =test_data.groupby('people_id')

frame_x=pd.DataFrame(people_group['date_x'].agg({'min_date_x':np.min}))
frame_y=pd.DataFrame(people_group['date_y'].agg({'min_date_y':np.min}))
frame_x.reset_index(level='people_id',inplace=True)
frame_y.reset_index(level='people_id',inplace=True)

frame=pd.merge(frame_x,frame_y,on='people_id')
frame['time_diff']=((frame.min_date_x -frame.min_date_y)/np.timedelta64(1,'D')).astype(int)
test_data=pd.merge(test_data,frame.loc[:,['people_id','time_diff']],on='people_id')

del people_group, frame

print ('')

train = train_data.sort_values(['people_id'], ascending=[1])
test = test_data.sort_values(['people_id'], ascending=[1])

train['business_days_delta'] = np.busday_count(train['date_y'].values.astype('<M8[D]'),
                                               train['date_x'].values.astype('<M8[D]'))
test['business_days_delta'] = np.busday_count(test['date_y'].values.astype('<M8[D]'),
                                              test['date_x'].values.astype('<M8[D]'))

print('Merging Features...')
activity_ids = test['activity_id']

print('Reading Data...')
N = train.shape[0]

print('Analyzing Data...')
train_y = train['outcome'].values
train.drop(['outcome'], inplace=True, axis=1)

train_columns = train.columns.values
test_columns = test.columns.values
features = list(set(train_columns) & set(test_columns))

X_train = train.sort_values(['people_id'], ascending=[1])
X_test = test[features]

X_train['outcome'] = train_y
X_train.to_csv(train_path, index=False)
X_test.to_csv(test_path, index=False)