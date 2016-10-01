import os
import pandas as pd
from heapq import nlargest
from operator import itemgetter
import ml_metrics as metrics
import numpy as np
import datetime
import operator
import math

def data_analysis():

    print ('')
    print ('Preparing Arrays...')
    print ('')

    f = open(os.path.expanduser('~/Desktop/Expedia/CV_Splits/split_train.csv'), "r")
    f.readline()

    data_analysis = dict()

    total = 0

    while 1:
        line = f.readline().strip()
        total += 1

        if line == '':
            break

        arr = line.split(",")

        site_name = arr[1]
        posa_continent = arr[2]
        user_location_country = arr[3]
        user_location_region = arr[4]
        user_location_city = arr[5]
        orig_destination_distance = arr[6]
        user_id = arr[7]
        is_mobile = arr[8]
        is_package = arr[9]
        channel = arr[10]

        srch_ci = arr[11]

        if srch_ci != 'nan':
            book_year = int(srch_ci[:4])
            book_month = int(srch_ci[5:7])
        else:
            book_year = int(arr[0][:4])
            book_month = int(arr[0][5:7])

        if srch_ci != 'nan':
            srch_ci_month = int(srch_ci[5:7])
        else:
            srch_ci_month = int(arr[0][5:7])

        srch_adults_cnt = arr[13]
        srch_children_cnt = arr[14]
        srch_rm_cnt = arr[15]
        srch_destination_id = arr[16]
        is_booking = float(arr[18])
        hotel_continent = arr[20]
        hotel_country = arr[21]
        hotel_market = arr[22]
        hotel_cluster = arr[23]

        if is_package == '1' and is_booking == 1 and srch_ci_month == 1 and hotel_market == '628':
           print (hotel_cluster)

data_analysis()

