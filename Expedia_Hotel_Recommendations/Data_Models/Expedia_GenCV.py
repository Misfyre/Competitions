import pandas as pd
import os

DATE_FEATURES = ['date_time', 'srch_ci', 'srch_co']
TRAIN_ONLY = ['is_booking', 'cnt']
TARGET = 'hotel_cluster'

NROWS = None

def has_year(df, year):
    return (df['date_time'].dt.year == year).any()

def has_booking_in_year(df, year):
    return ((df['date_time'].dt.year == year) & (df['is_booking'] == 1)).any()

def load_cv():
    try:
        df_2014 = pd.read_csv(os.path.expanduser('~/Desktop/Expedia/CV_Splits/split_train.csv'), parse_dates=DATE_FEATURES)
        bookings_2013 = pd.read_csv(os.path.expanduser('~/Desktop/Expedia/CV_Splits/split_test.csv'), parse_dates=DATE_FEATURES)

    # OSError is because of the Kaggle infrastructure.
    # Set to IOError for python 2 and FileNotFoundError for python 3.

    except OSError:

        print('CV not found. Building CV datasets.')
        df = pd.read_csv(os.path.expanduser('~/Desktop/Expedia/Data/train.csv'), parse_dates=DATE_FEATURES, nrows=NROWS)
        pd.to_datetime(df['date_time'])
        grouped = df.groupby('user_id')

        good_users = grouped.filter(lambda x: (has_year(x, 2014) and has_booking_in_year(x, 2013)))

        df_2014 = good_users[good_users['date_time'].dt.year == 2014]
        df_2013 = good_users[good_users['date_time'].dt.year == 2013]

        bookings_2013 = df_2013[df_2013['is_booking'] == 1].drop(TRAIN_ONLY, axis=1)
        bookings_2013.insert(0, 'id', range(len(bookings_2013)))

        df_2014.to_csv(os.path.expanduser('~/Desktop/Expedia/CV_Splits/split_train.csv'), index=False)
        bookings_2013.to_csv(os.path.expanduser('~/Desktop/Expedia/CV_Splits/split_test.csv'), index=False)

    print (df_2014)
    return df_2014, bookings_2013.drop(TARGET, axis=1), bookings_2013[TARGET]

if __name__ == '__main__':
    load_cv()
