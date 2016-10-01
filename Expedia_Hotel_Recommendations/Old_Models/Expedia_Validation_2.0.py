import os
import pandas as pd
from heapq import nlargest
from operator import itemgetter
import ml_metrics as metrics
from datetime import date
import numpy as np
import math

def month_to_season(month, day):

    if month == 12 and day >= 21:
        season = "Winter"
        return season

    elif month in (1,2):
        season = "Winter"
        return season

    elif month == 3 and day < 21:
        season = "Winter"
        return season

    elif month == 3 and day >= 21:
        season = "Spring"
        return season

    elif month in (4,5):
        season = "Spring"
        return season

    elif month == 6 and day < 21:
        season = "Spring"
        return season

    elif month == 6 and day >= 21:
        season = "Summer"
        return season

    elif month in (7,8):
        season = "Summer"
        return season

    elif month == 9 and day < 21:
        season = "Summer"
        return season

    elif month == 9 and day >= 21:
        season = "Fall"
        return season

    elif month in (10,11):
        season = "Fall"
        return season

    elif month == 12 and day < 21:
        season = "Fall"
        return season

def prepare_arrays_match():

    print ('')
    print ('Preparing Arrays...')
    print ('')

    f = open(os.path.expanduser('~/Desktop/Expedia/CV_Splits/split_train.csv'), "r")
    f.readline()

    best_hotels_od_ulc = dict()
    best_hotels_uid_miss = dict()
    best_hotels_search_dest = dict()
    best_hotels_country = dict()

    popular_hotel_cluster = dict()

    best_s00 = dict()
    best_s01 = dict()
    best_s02 = dict()

    total = 0

    while 1:

        line = f.readline().strip()
        total += 1

        if total % 2000000 == 0:
            print('Read {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")

        if arr[11] != 'nan':
            book_year = int(arr[11][:4])
            book_month = int(arr[11][5:7])
        else:
            book_year = int(arr[0][:4])
            book_month = int(arr[0][5:7])

        if book_month<1 or book_month>12 or book_year<2012 or book_year>2015:
            continue

        user_location_city = arr[5]
        orig_destination_distance = arr[6]

        user_id = arr[7]
        srch_destination_id = arr[16]

        srch_ci = arr[11]
        srch_co = arr[12]

        if srch_ci != 'nan':
            srch_ci_month = int(srch_ci[5:7])
        else:
            srch_ci_month = ''

        if srch_co != 'nan':
            srch_co_month = int(srch_co[5:7])
        else:
            srch_co_month = ''

        hotel_country = arr[21]
        hotel_market = arr[22]
        is_booking = float(arr[18])
        hotel_cluster = arr[23]

        append_0 = ((book_year - 2012)*12 + (book_month - 12))

        if not (append_0>0 and append_0<=36):
            continue

        append_1 = pow(append_0, 0.5) * append_0 * (3 + 17.60*is_booking)
        append_2 = 3 * math.floor(((book_month+1)%12) / 4) + 5.56*is_booking

        if user_location_city != '' and orig_destination_distance != '' and user_id !='' and srch_destination_id != '' and hotel_country != '' and is_booking==1:
            s00 = (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
            if s00 in best_s00:
                if hotel_cluster in best_s00[s00]:
                    best_s00[s00][hotel_cluster] += append_0
                else:
                    best_s00[s00][hotel_cluster] = append_0
            else:
                best_s00[s00] = dict()
                best_s00[s00][hotel_cluster] = append_0

        if user_location_city != '' and orig_destination_distance != '' and user_id !='' and srch_destination_id != '' and is_booking==1:
            s01 = (user_id, srch_destination_id, hotel_country, hotel_market)
            if s01 in best_s01:
                if hotel_cluster in best_s01[s01]:
                    best_s01[s01][hotel_cluster] += append_0
                else:
                    best_s01[s01][hotel_cluster] = append_0
            else:
                best_s01[s01] = dict()
                best_s01[s01][hotel_cluster] = append_0

        if user_id != '' and srch_destination_id != '' and srch_ci_month != '' and srch_co_month != '' and is_booking==1:
            s02 = (user_id, srch_destination_id, srch_ci_month, srch_co_month)
            if s02 in best_s02:
                if hotel_cluster in best_s02[s02]:
                    best_s02[s02][hotel_cluster] += append_0
                else:
                    best_s02[s02][hotel_cluster] = append_0
            else:
                best_s02[s02] = dict()
                best_s02[s02][hotel_cluster] = append_0

        if user_location_city != '' and orig_destination_distance == '' and user_id !='' and srch_destination_id != '' and hotel_country != '' and is_booking==1:
            s0 = (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
            if s0 in best_hotels_uid_miss:
                if hotel_cluster in best_hotels_uid_miss[s0]:
                    best_hotels_uid_miss[s0][hotel_cluster] += append_0
                else:
                    best_hotels_uid_miss[s0][hotel_cluster] = append_0
            else:
                best_hotels_uid_miss[s0] = dict()
                best_hotels_uid_miss[s0][hotel_cluster] = append_0

        if user_location_city != '' and orig_destination_distance != '':
            s1 = (user_location_city, orig_destination_distance)

            if s1 in best_hotels_od_ulc:
                if hotel_cluster in best_hotels_od_ulc[s1]:
                    best_hotels_od_ulc[s1][hotel_cluster] += append_0
                else:
                    best_hotels_od_ulc[s1][hotel_cluster] = append_0
            else:
                best_hotels_od_ulc[s1] = dict()
                best_hotels_od_ulc[s1][hotel_cluster] = append_0

        if srch_destination_id != '' and hotel_country != '' and hotel_market != '':
            s2 = (srch_destination_id,hotel_country,hotel_market)
            if s2 in best_hotels_search_dest:
                if hotel_cluster in best_hotels_search_dest[s2]:
                    best_hotels_search_dest[s2][hotel_cluster] += append_1
                else:
                    best_hotels_search_dest[s2][hotel_cluster] = append_1
            else:
                best_hotels_search_dest[s2] = dict()
                best_hotels_search_dest[s2][hotel_cluster] = append_1

        if hotel_country != '':
            s3 = (hotel_country)
            if s3 in best_hotels_country:
                if hotel_cluster in best_hotels_country[s3]:
                    best_hotels_country[s3][hotel_cluster] += append_2
                else:
                    best_hotels_country[s3][hotel_cluster] = append_2
            else:
                best_hotels_country[s3] = dict()
                best_hotels_country[s3][hotel_cluster] = append_2

        if hotel_cluster in popular_hotel_cluster:
            popular_hotel_cluster[hotel_cluster] += append_0
        else:
            popular_hotel_cluster[hotel_cluster] = append_0

    f.close()
    return best_s00,best_s01, best_s02, best_hotels_country, best_hotels_od_ulc, best_hotels_uid_miss, best_hotels_search_dest, popular_hotel_cluster

def gen_submission(best_s00, best_s01, best_s02, best_hotels_country, best_hotels_search_dest, best_hotels_od_ulc,
                   best_hotels_uid_miss, popular_hotel_cluster):

    print ('')
    path= os.path.expanduser('~/Desktop/Expedia/CVs/cv_test.csv')
    out = open(path, "w")
    f = open(os.path.expanduser('~/Desktop/Expedia/CV_Splits/split_test.csv'), "r")
    f.readline()

    total = 0
    total1 = 0
    total2 = 0
    total3 = 0
    total4 = 0
    total5 = 0
    total6 = 0
    total7 = 0

    out.write("id,hotel_cluster\n")
    topclasters = nlargest(5, sorted(popular_hotel_cluster.items()), key=itemgetter(1))

    while 1:
        line = f.readline().strip()
        total += 1

        if total % 200000 == 0:
            print('Write {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")

        id = arr[0]
        user_location_city = arr[6]
        orig_destination_distance = arr[7]

        user_id = arr[8]
        srch_destination_id = arr[17]

        srch_ci = arr[12]
        srch_co = arr[13]

        if srch_ci != 'nan':
            srch_ci_month = int(srch_ci[5:7])
        else:
            srch_ci_month = ''

        if srch_co != 'nan':
            srch_co_month = int(srch_co[5:7])
        else:
            srch_co_month = ''

        hotel_country = arr[20]
        hotel_market = arr[21]

        out.write(str(id) + ',')
        filled = []

        s1 = (user_location_city, orig_destination_distance)
        if s1 in best_hotels_od_ulc:
            d = best_hotels_od_ulc[s1]
            topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                total1 += 1

        if orig_destination_distance == '':
            s2 = (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
            if s2 in best_hotels_uid_miss:
                d = best_hotels_uid_miss[s2]
                topitems = nlargest(4, sorted(d.items()), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])
                    total2 += 1

        if orig_destination_distance == '':
            s02 = (user_id, srch_destination_id, srch_ci_month, srch_co_month)
            if s02 in best_s02:
                d = best_s02[s02]
                topitems = nlargest(4, sorted(d.items()), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])
                    total3 += 1

        s3 = (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
        s4 = (user_id, srch_destination_id, hotel_country, hotel_market)
        if s4 in best_s01 and s3 not in best_s00:
            d = best_s01[s4]
            topitems = nlargest(4, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                total4 += 1


        s5 = (srch_destination_id,hotel_country,hotel_market)
        if s5 in best_hotels_search_dest:
            d = best_hotels_search_dest[s5]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                total5 += 1

        s6 = (hotel_country)
        if s6 in best_hotels_country:
            d = best_hotels_country[s6]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                total6 += 1

        for i in range(len(topclasters)):
            if topclasters[i][0] in filled:
                continue
            if len(filled) == 5:
                break
            out.write(' ' + topclasters[i][0])
            filled.append(topclasters[i][0])
            total7 += 1

        out.write("\n")
    out.close()

    print ('')

    print ('Total 1: {} ...'.format(total1))
    print ('Total 2: {} ...'.format(total2))
    print ('Total 3: {} ...'.format(total3))
    print ('Total 4: {} ...'.format(total4))
    print ('Total 5: {} ...'.format(total5))
    print ('Total 6: {} ...'.format(total6))
    print ('Total 7: {} ...'.format(total7))

    print ('')
    print ('Loading Data...')

    cv_submission_path = os.path.expanduser('~/Desktop/Expedia/CVs/cv_test.csv')
    split_test_path = os.path.expanduser('~/Desktop/Expedia/CV_Splits/split_test.csv')

    submission_cv = pd.read_csv(cv_submission_path, usecols = ['hotel_cluster'])
    split_validation = pd.read_csv(split_test_path, usecols = ['hotel_cluster'], dtype = {'hotel_cluster':np.int16})

    print ('Data Loaded...')

    preds = []

    for i in range(submission_cv.shape[0]):
        arr = submission_cv.hotel_cluster[i].split(" ")
        arr = list(arr[1:10])
        arr = list(map(int, arr))
        preds.append(arr)

    target_test = [[l] for l in split_validation["hotel_cluster"]]

    print ('')
    print ('Score:',metrics.mapk(target_test, preds, k=5))


best_s00, best_s01, best_s02, best_hotels_country, best_hotels_od_ulc, best_hotels_uid_miss, best_hotels_search_dest, popular_hotel_cluster = prepare_arrays_match()
gen_submission(best_s00, best_s01, best_s02, best_hotels_country, best_hotels_search_dest, best_hotels_od_ulc, best_hotels_uid_miss, popular_hotel_cluster)
