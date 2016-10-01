import os
import pandas as pd
from heapq import nlargest
from operator import itemgetter
import ml_metrics as metrics
import numpy as np
import math

train_path = os.path.expanduser('~/Desktop/Expedia/Data/train.csv')
test_path = os.path.expanduser('~/Desktop/Expedia/Data/test.csv')
cv_path = os.path.expanduser('~/Desktop/Expedia/Submission/submission.csv')

def prepare_arrays_test(train_path):

    print ('')
    print ('Generating Kaggle Submission...')
    print ('Preparing Arrays...')
    print ('')

    f = open(train_path, "r")
    f.readline()

    best_hotels_od_ulc = dict()
    best_hotels_search_dest = dict()
    best_hotels_user_ci = dict()
    best_hotels_city_ci = dict()
    best_hotels_uid_miss = dict()
    best_hotels_country = dict()

    best_s00 = dict()
    best_s01 = dict()

    popular_hotel_cluster = dict()

    total = 0

    while 1:

        line = f.readline().strip()
        total += 1

        if total % 5000000 == 0:
            print('Read {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")

        user_location_city = arr[5]
        orig_destination_distance = arr[6]
        user_id = arr[7]

        srch_ci = arr[11]

        if srch_ci != '':
            book_year = int(srch_ci[:4])
            book_month = int(srch_ci[5:7])
            srch_ci_month = int(srch_ci[5:7])

        else:
            book_year = int(arr[0][:4])
            book_month = int(arr[0][5:7])
            srch_ci_month = int(arr[0][5:7])

        srch_destination_id = arr[16]
        is_booking = float(arr[18])
        hotel_country = arr[21]
        hotel_market = arr[22]
        hotel_cluster = arr[23]

        append_0 = ((book_year - 2012)*12 + (book_month - 12))
        if not (append_0>0 and append_0<=36):
            continue

        append_1 = pow(math.log(append_0), 1.35) * (-0.1+0.95*pow(append_0, 1.46)) * (3.5 + 17.6*is_booking)
        append_2 = 3 + 5.56*is_booking

        ### best_s00

        if user_location_city != '' and orig_destination_distance != '' and user_id !='' and srch_destination_id != '' and hotel_country != '':
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

        if user_location_city != '' and orig_destination_distance != '' and user_id !='' and srch_destination_id != '':
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
            hsh = hash('user_location_city' + str(user_location_city) + 'orig_destination_distance' + str(orig_destination_distance))

            if hsh in best_hotels_od_ulc:
                if hotel_cluster in best_hotels_od_ulc[hsh]:
                    best_hotels_od_ulc[hsh][hotel_cluster] += append_0
                else:
                    best_hotels_od_ulc[hsh][hotel_cluster] = append_0
            else:
                best_hotels_od_ulc[hsh] = dict()
                best_hotels_od_ulc[hsh][hotel_cluster] = append_0

        ### best_hotels_uid_miss

        if user_location_city != '' and orig_destination_distance == '' and user_id !='' and srch_destination_id != '' and hotel_country != '':
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

        if user_location_city != '' and srch_destination_id != '' and srch_ci_month != '':
            hsh = hash('user_id' + str(user_id) + 'srch_destination_id' + str(srch_destination_id) + 'srch_ci_month' + str(srch_ci_month) + 'hotel_country' + str(hotel_country) + 'hotel_market' + str(hotel_market))

            if hsh in best_hotels_user_ci:
                if hotel_cluster in best_hotels_user_ci[hsh]:
                    best_hotels_user_ci[hsh][hotel_cluster] += append_0
                else:
                    best_hotels_user_ci[hsh][hotel_cluster] = append_0
            else:
                best_hotels_user_ci[hsh] = dict()
                best_hotels_user_ci[hsh][hotel_cluster] = append_0

        ### best_hotels_city_ci

        if user_location_city != '' and srch_destination_id != '' and srch_ci_month != '':
            hsh = hash('user_id' + str(user_id) + 'user_location_city' + str(user_location_city) + 'srch_destination_id' + str(srch_destination_id) + 'srch_ci_month' + str(srch_ci_month) + 'hotel_country' + str(hotel_country) + 'hotel_market' + str(hotel_market))
            
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
            hsh = hash('hotel_market' + str(hotel_market))

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


def gen_submission_test(test_path, cv_path, best_s00, best_s01, best_hotels_uid_miss, best_hotels_od_ulc, best_hotels_search_dest, best_hotels_user_ci, best_hotels_city_ci, best_hotels_country, popular_hotel_cluster):

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

        if total % 500000 == 0:
            print('Write {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")

        user_location_city = arr[6]
        orig_destination_distance = arr[7]
        user_id = arr[8]

        srch_ci = arr[12]

        if srch_ci != '':
            srch_ci_month = int(srch_ci[5:7])

        else:
            srch_ci_month = int(arr[1][5:7])

        srch_destination_id = arr[17]

        hotel_country = arr[20]
        hotel_market = arr[21]

        out.write(str(id) + ',')
        filled = []

        hsh = hash('user_location_city' + str(user_location_city) + 'orig_destination_distance' + str(orig_destination_distance))
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

        hsh = hash('user_id' + str(user_id) + 'user_location_city' + str(user_location_city) + 'srch_destination_id' + str(srch_destination_id) + 'srch_ci_month' + str(srch_ci_month) + 'hotel_country' + str(hotel_country) + 'hotel_market' + str(hotel_market))
        hsh1 = hash('user_id' + str(user_id) + 'srch_destination_id' + str(srch_destination_id) + 'srch_ci_month' + str(srch_ci_month) + 'hotel_country' + str(hotel_country) + 'hotel_market' + str(hotel_market))
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

        hsh = hash('hotel_market' + str(hotel_market))
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
    print ('Completed...')

best_s00, best_s01, best_hotels_uid_miss, best_hotels_od_ulc, best_hotels_search_dest, best_hotels_user_ci, best_hotels_city_ci, best_hotels_country, popular_hotel_cluster = prepare_arrays_test(train_path)
gen_submission_test(test_path, cv_path, best_s00, best_s01, best_hotels_uid_miss, best_hotels_od_ulc, best_hotels_search_dest, best_hotels_user_ci, best_hotels_city_ci, best_hotels_country, popular_hotel_cluster)
