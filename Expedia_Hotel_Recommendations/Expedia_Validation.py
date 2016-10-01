import os
import pandas as pd
from heapq import nlargest
from operator import itemgetter
import ml_metrics as metrics
import numpy as np
import math

train_path = os.path.expanduser('~/Desktop/Expedia/CV_Splits/split_train.csv')
test_path = os.path.expanduser('~/Desktop/Expedia/CV_Splits/split_test.csv')
cv_path = os.path.expanduser('~/Desktop/Expedia/CVs/cv_test.csv')

def prepare_arrays_validation(train_path):

    print ('')
    print ('Generating Validation Score...')
    print ('Preparing Arrays...')
    print ('')

    f = open(train_path, "r")
    f.readline()

    best_hotels_od_ulc = dict()
    best_hotels_search_dest = dict()
    best_hotels_uid_miss = dict()
    best_hotels_country = dict()

    best_hotels_user_ci = dict()
    best_hotels_city_ci = dict()

    best_s00 = dict()
    best_s01 = dict()

    popular_hotel_cluster = dict()

    total = 0

    while 1:

        line = f.readline().strip()
        total += 1

        if total % 2000000 == 0:
            print('Read {} lines...'.format(total))

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
            srch_ci_month = int(srch_ci[5:7])

        else:
            book_year = int(arr[0][:4])
            book_month = int(arr[0][5:7])
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

        append_0 = ((book_year - 2012)*12 + (book_month - 12))
        if not (append_0>0 and append_0<=36):
            continue

        append_1 = pow(math.log(append_0), 1.35) * (-0.1+0.95*pow(append_0, 1.46)) * (3.5 + 17.6*is_booking)
        append_2 = 3 + 5.56*is_booking

        ### best_s00

        if user_location_city != '' and orig_destination_distance != '' and user_id !='' and srch_destination_id != '' and hotel_country != '' and is_booking == 1:
            hsh = hash('user_id' + str(user_id) + 'user_location_city' + str(user_location_city) + 'srch_destination_id' + str(srch_destination_id) + 'hotel_country' + str(hotel_country) + 'hotel_market' + str(hotel_market))

            if hsh in best_s00:
                if hotel_cluster in best_s00[hsh]:
                    best_s00[hsh][hotel_cluster] += append_0
                else:
                    best_s00[hsh][hotel_cluster] = append_0
            else:
                best_s00[hsh] = dict()
                best_s00[hsh][hotel_cluster] = append_0

        ### best_s01

        if user_location_city != '' and orig_destination_distance != '' and user_id !='' and srch_destination_id != '' and is_booking == 1:
            hsh = hash('user_id' + str(user_id) + 'srch_destination_id' + str(srch_destination_id) + 'hotel_country' + str(hotel_country) + 'hotel_market' + str(hotel_market))

            if hsh in best_s01:
                if hotel_cluster in best_s01[hsh]:
                    best_s01[hsh][hotel_cluster] += append_0
                else:
                    best_s01[hsh][hotel_cluster] = append_0
            else:
                best_s01[hsh] = dict()
                best_s01[hsh][hotel_cluster] = append_0

        ### best_hotels_od_ulc

        if user_location_city != '' and orig_destination_distance != '':
            hsh = hash('user_location_city' + str(user_location_city) + 'orig_destination_distance' + str(orig_destination_distance) + 'hotel_market' + str(hotel_market))

            if hsh in best_hotels_od_ulc:
                if hotel_cluster in best_hotels_od_ulc[hsh]:
                    best_hotels_od_ulc[hsh][hotel_cluster] += append_0
                else:
                    best_hotels_od_ulc[hsh][hotel_cluster] = append_0
            else:
                best_hotels_od_ulc[hsh] = dict()
                best_hotels_od_ulc[hsh][hotel_cluster] = append_0

        ### best_hotels_uid_miss

        if user_location_city != '' and orig_destination_distance == '' and user_id !='' and srch_destination_id != '' and hotel_country != '' and is_booking == 1:
            hsh = hash('user_id' + str(user_id) + 'user_location_city' + str(user_location_city) + 'srch_destination_id' + str(srch_destination_id) + 'hotel_country' + str(hotel_country) + 'hotel_market' + str(hotel_market))

            if hsh in best_hotels_uid_miss:
                if hotel_cluster in best_hotels_uid_miss[hsh]:
                    best_hotels_uid_miss[hsh][hotel_cluster] += append_0
                else:
                    best_hotels_uid_miss[hsh][hotel_cluster] = append_0
            else:
                best_hotels_uid_miss[hsh] = dict()
                best_hotels_uid_miss[hsh][hotel_cluster] = append_0

        ### best_hotels_search_dest

        if srch_destination_id != '' and hotel_country != '' and hotel_market != '':
            hsh = hash('srch_destination_id' + str(srch_destination_id) + 'hotel_country' + str(hotel_country) + 'hotel_market' + str(hotel_market))

            if hsh in best_hotels_search_dest:
                if hotel_cluster in best_hotels_search_dest[hsh]:
                    best_hotels_search_dest[hsh][hotel_cluster] += append_1
                else:
                    best_hotels_search_dest[hsh][hotel_cluster] = append_1
            else:
                best_hotels_search_dest[hsh] = dict()
                best_hotels_search_dest[hsh][hotel_cluster] = append_1

        ### best_hotels_user_ci

        if user_location_city != '' and hotel_market != '' and srch_ci_month != '' and is_booking == 1:
            hsh = hash('user_id' + str(user_id) + 'hotel_market' + str(hotel_market) + 'srch_ci_month' + str(srch_ci_month))

            if hsh in best_hotels_user_ci:
                if hotel_cluster in best_hotels_user_ci[hsh]:
                    best_hotels_user_ci[hsh][hotel_cluster] += append_0
                else:
                    best_hotels_user_ci[hsh][hotel_cluster] = append_0
            else:
                best_hotels_user_ci[hsh] = dict()
                best_hotels_user_ci[hsh][hotel_cluster] = append_0

        ### best_hotels_city_ci

        if user_location_city != '' and hotel_market != '' and srch_ci_month != '' and is_booking == 1:
            hsh = hash('user_id' + str(user_id) + 'user_location_city' + str(user_location_city) + 'hotel_market' + str(hotel_market) + 'srch_ci_month' + str(srch_ci_month))

            if hsh in best_hotels_city_ci:
                if hotel_cluster in best_hotels_city_ci[hsh]:
                    best_hotels_city_ci[hsh][hotel_cluster] += append_0
                else:
                    best_hotels_city_ci[hsh][hotel_cluster] = append_0
            else:
                best_hotels_city_ci[hsh] = dict()
                best_hotels_city_ci[hsh][hotel_cluster] = append_0

        ### best_hotels_country

        if hotel_market != '':
            hsh = hash('hotel_market' + str(hotel_market) + 'is_package' + str(is_package))

            if hsh in best_hotels_country:
                if hotel_cluster in best_hotels_country[hsh]:
                    best_hotels_country[hsh][hotel_cluster] += append_2
                else:
                    best_hotels_country[hsh][hotel_cluster] = append_2
            else:
                best_hotels_country[hsh] = dict()
                best_hotels_country[hsh][hotel_cluster] = append_2

        ### popular_hotel_cluster

        if hotel_cluster in popular_hotel_cluster:
            popular_hotel_cluster[hotel_cluster] += append_0
        else:
            popular_hotel_cluster[hotel_cluster] = append_0

    f.close()
    return best_s00, best_s01, best_hotels_uid_miss, best_hotels_od_ulc, best_hotels_search_dest, best_hotels_user_ci, best_hotels_city_ci, best_hotels_country, popular_hotel_cluster


def gen_submission_validation(test_path, cv_path, best_s00, best_s01, best_hotels_uid_miss, best_hotels_od_ulc, best_hotels_search_dest, best_hotels_user_ci, best_hotels_city_ci, best_hotels_country, popular_hotel_cluster):

    print ('')
    path= cv_path
    out = open(path, "w")
    f = open(test_path, "r")
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

        if total % 300000 == 0:
            print('Write {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")

        site_name = arr[2]
        posa_continent = arr[3]
        user_location_country = arr[4]
        user_location_region = arr[5]
        user_location_city = arr[6]
        orig_destination_distance = arr[7]
        user_id = arr[8]
        is_mobile = arr[9]
        is_package = arr[10]
        channel = arr[11]

        srch_ci = arr[12]

        if srch_ci != 'nan':
            book_year = int(srch_ci[:4])
            book_month = int(srch_ci[5:7])
            srch_ci_month = int(srch_ci[5:7])

        else:
            book_year = int(arr[0][:4])
            book_month = int(arr[0][5:7])
            srch_ci_month = int(arr[0][5:7])

        srch_adults_cnt = arr[14]
        srch_children_cnt = arr[15]
        srch_rm_cnt = arr[16]
        srch_destination_id = arr[17]

        hotel_country = arr[20]
        hotel_market = arr[21]

        out.write(str(id) + ',')
        filled = []

        hsh = hash('user_location_city' + str(user_location_city) + 'orig_destination_distance' + str(orig_destination_distance) + 'hotel_market' + str(hotel_market))
        if hsh in best_hotels_od_ulc:
            d = best_hotels_od_ulc[hsh]
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
            hsh = hash('user_id' + str(user_id) + 'user_location_city' + str(user_location_city) + 'srch_destination_id' + str(srch_destination_id) + 'hotel_country' + str(hotel_country) + 'hotel_market' + str(hotel_market))
            if hsh in best_hotels_uid_miss:
                d = best_hotels_uid_miss[hsh]
                topitems = nlargest(4, sorted(d.items()), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])
                    total2 += 1

        hsh = hash('user_id' + str(user_id) + 'user_location_city' + str(user_location_city) + 'srch_destination_id' + str(srch_destination_id) + 'hotel_country' + str(hotel_country) + 'hotel_market' + str(hotel_market))
        hsh1 = hash('user_id' + str(user_id) + 'srch_destination_id' + str(srch_destination_id) + 'hotel_country' + str(hotel_country) + 'hotel_market' + str(hotel_market))
        if hsh1 in best_s01 and hsh not in best_s00:
            d = best_s01[hsh1]
            topitems = nlargest(4, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                total3 += 1

        hsh = hash('user_id' + str(user_id) + 'user_location_city' + str(user_location_city) + 'hotel_market' + str(hotel_market) + 'srch_ci_month' + str(srch_ci_month))
        hsh1 = hash('user_id' + str(user_id) + 'hotel_market' + str(hotel_market) + 'srch_ci_month' + str(srch_ci_month))
        if hsh1 in best_hotels_user_ci and hsh not in best_hotels_city_ci:
            d = best_hotels_user_ci[hsh1]
            topitems = nlargest(4, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                total4 += 1

        hsh = hash('srch_destination_id' + str(srch_destination_id) + 'hotel_country' + str(hotel_country) + 'hotel_market' + str(hotel_market))
        if hsh in best_hotels_search_dest:
            d = best_hotels_search_dest[hsh]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                total5 += 1

        hsh = hash('hotel_market' + str(hotel_market) + 'is_package' + str(is_package))
        if hsh in best_hotels_country:
            d = best_hotels_country[hsh]
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

best_s00, best_s01, best_hotels_uid_miss, best_hotels_od_ulc, best_hotels_search_dest, best_hotels_user_ci, best_hotels_city_ci, best_hotels_country, popular_hotel_cluster = prepare_arrays_validation(train_path)
gen_submission_validation(test_path, cv_path, best_s00, best_s01, best_hotels_uid_miss, best_hotels_od_ulc, best_hotels_search_dest, best_hotels_user_ci, best_hotels_city_ci, best_hotels_country, popular_hotel_cluster)
