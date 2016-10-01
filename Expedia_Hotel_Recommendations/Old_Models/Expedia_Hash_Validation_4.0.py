import os
import pandas as pd
from heapq import nlargest
from operator import itemgetter
import ml_metrics as metrics
import numpy as np
import math

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

    destination_clusters = dict()

    total = 0

    while 1:
        line = f.readline().strip()
        total += 1

        if total % 2000000 == 0:
            print('Read {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")

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

        if book_month<1 or book_month>12 or book_year<2012 or book_year>2015:
            continue

        user_location_city = arr[5]
        orig_destination_distance = arr[6]
        user_id = arr[7]
        srch_destination_id = arr[16]
        hotel_country = arr[21]
        hotel_market = arr[22]
        is_booking = float(arr[18])
        hotel_cluster = arr[23]

        append_0 = ((book_year - 2012)*12 + (book_month - 12))
        if not (append_0>0 and append_0<=36):
            continue

        append_1 = pow(append_0, 0.45) * append_0 * (3.5 + 17.60*is_booking)
        append_2 = 3 + 5.56*is_booking

        ### best_s00

        if user_location_city != '' and orig_destination_distance != '' and user_id !='' and srch_destination_id != '' and hotel_country != '' and is_booking==1:
            hsh = hash('user_id' + str(user_id) + 'user_location_city' + str(user_location_city) + 'srch_destination_id'
                       + str(srch_destination_id) + 'hotel_country' + str(hotel_country) + 'hotel_market' + str(hotel_market))

            if hsh in best_s00:
                if hotel_cluster in best_s00[hsh]:
                    best_s00[hsh][hotel_cluster] += append_0
                else:
                    best_s00[hsh][hotel_cluster] = append_0
            else:
                best_s00[hsh] = dict()
                best_s00[hsh][hotel_cluster] = append_0

        ### best_s01

        if user_location_city != '' and orig_destination_distance != '' and user_id !='' and srch_destination_id != '' and is_booking==1:
            hsh = hash('user_id' + str(user_id) + 'srch_destination_id' + str(srch_destination_id)
                   + 'hotel_country' + str(hotel_country) + 'hotel_market' + str(hotel_market))

            if hsh in best_s01:
                if hotel_cluster in best_s01[hsh]:
                    best_s01[hsh][hotel_cluster] += append_0
                else:
                    best_s01[hsh][hotel_cluster] = append_0
            else:
                best_s01[hsh] = dict()
                best_s01[hsh][hotel_cluster] = append_0

        ### best_s02

        if user_id != '' and srch_destination_id != '' and srch_ci_month != '':
            hsh = hash('user_id' + str(user_id) + 'srch_destination_id' + str(srch_destination_id) + 'srch_ci_month' + str(srch_ci_month))

            if hsh in best_s02:
                if hotel_cluster in best_s02[hsh]:
                    best_s02[hsh][hotel_cluster] += append_0
                else:
                    best_s02[hsh][hotel_cluster] = append_0
            else:
                best_s02[hsh] = dict()
                best_s02[hsh][hotel_cluster] = append_0

        ### best_hotels_uid_miss

        if user_location_city != '' and orig_destination_distance == '' and user_id !='' and srch_destination_id != '' and hotel_country != '' and is_booking==1:
            hsh = hash('user_id' + str(user_id) + 'user_location_city' + str(user_location_city) + 'srch_destination_id'
                    + str(srch_destination_id) + 'hotel_country' + str(hotel_country) + 'hotel_market' + str(hotel_market))

            if hsh in best_hotels_uid_miss:
                if hotel_cluster in best_hotels_uid_miss[hsh]:
                    best_hotels_uid_miss[hsh][hotel_cluster] += append_0
                else:
                    best_hotels_uid_miss[hsh][hotel_cluster] = append_0
            else:
                best_hotels_uid_miss[hsh] = dict()
                best_hotels_uid_miss[hsh][hotel_cluster] = append_0

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

        ### best_hotels_search_dest

        if srch_destination_id != '' and hotel_country != '' and hotel_market != '':
            hsh = hash('srch_destination_id' + str(srch_destination_id) + 'hotel_country' + str(hotel_country)
                       + 'hotel_market' + str(hotel_market))

            if hsh in best_hotels_search_dest:
                if hotel_cluster in best_hotels_search_dest[hsh]:
                    best_hotels_search_dest[hsh][hotel_cluster] += append_1
                else:
                    best_hotels_search_dest[hsh][hotel_cluster] = append_1
            else:
                best_hotels_search_dest[hsh] = dict()
                best_hotels_search_dest[hsh][hotel_cluster] = append_1

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

        ### destination_clusters

        if srch_destination_id != '':
            destinations = srch_destination_id
            if destinations in destination_clusters:
                if hotel_cluster in destination_clusters[destinations]:
                    continue
                else:
                    destination_clusters[destinations].append(hotel_cluster)
            else:
                destination_clusters[destinations] = []
                destination_clusters[destinations].append(hotel_cluster)

    f.close()
    return best_s00,best_s01, best_s02, best_hotels_country, best_hotels_od_ulc, best_hotels_uid_miss, best_hotels_search_dest, popular_hotel_cluster, destination_clusters

def gen_submission(best_s00,best_s01, best_s02, best_hotels_country, best_hotels_od_ulc, best_hotels_uid_miss, best_hotels_search_dest, popular_hotel_cluster, destination_clusters):

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

    total6a = 0
    total6b = 0
    total6c = 0

    total7a = 0
    total7b = 0
    total7c = 0

    total8 = 0

    out.write("id,hotel_cluster\n")
    topclasters = sorted(popular_hotel_cluster.items(), key=itemgetter(1))

    while 1:

        line = f.readline().strip()
        total += 1

        if total % 100000 == 0:
            print('Write {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")

        srch_ci = arr[12]

        if srch_ci != 'nan':
            srch_ci_month = int(srch_ci[5:7])
        else:
            srch_ci_month = int(arr[1][5:7])

        id = arr[0]
        user_location_city = arr[6]
        orig_destination_distance = arr[7]
        user_id = arr[8]
        srch_destination_id = arr[17]
        hotel_country = arr[20]
        hotel_market = arr[21]

        out.write(str(id) + ',')
        filled = []

        s1 = hash('user_location_city' + str(user_location_city) + 'orig_destination_distance' + str(orig_destination_distance))
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
            s2 = hash('user_id' + str(user_id) + 'user_location_city' + str(user_location_city) + 'srch_destination_id' + str(srch_destination_id) + 'hotel_country' + str(hotel_country) + 'hotel_market' + str(hotel_market))
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

        s3 = hash('user_id' + str(user_id) + 'user_location_city' + str(user_location_city) + 'srch_destination_id' + str(srch_destination_id) + 'hotel_country' + str(hotel_country) + 'hotel_market' + str(hotel_market))
        s4 = hash('user_id' + str(user_id) + 'srch_destination_id' + str(srch_destination_id) + 'hotel_country' + str(hotel_country) + 'hotel_market' + str(hotel_market))
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
                total3 += 1

        s5 = hash('user_id' + str(user_id) + 'srch_destination_id' + str(srch_destination_id) + 'srch_ci_month' + str(srch_ci_month))
        if s5 in best_s02:
            d = best_s02[s5]
            topitems = nlargest(4, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                total4 += 1

        s6 = hash('srch_destination_id' + str(srch_destination_id) + 'hotel_country' + str(hotel_country) + 'hotel_market' + str(hotel_market))
        if s6 in best_hotels_search_dest:
            d = best_hotels_search_dest[s6]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                total5 += 1

        s7 = hash('hotel_market' + str(hotel_market))
        if s7 in best_hotels_country:
            d = best_hotels_country[s7]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if srch_destination_id in destination_clusters:
                    if topitems[i][0] in destination_clusters[srch_destination_id]:
                        if topitems[i][0] in filled:
                            continue
                        if len(filled) == 5:
                            break
                        out.write(' ' + topitems[i][0])
                        filled.append(topitems[i][0])
                        total6a += 1
                    else:
                        total6b += 1
                else:
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])
                    total6c += 1

        for i in range(len(topclasters)):
            if srch_destination_id in destination_clusters:
                if topclasters[i][0] in destination_clusters[srch_destination_id]:
                    if topclasters[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topclasters[i][0])
                    filled.append(topclasters[i][0])
                    total7a += 1
                else:
                    total7b += 1
            else:
                if topclasters[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topclasters[i][0])
                filled.append(topclasters[i][0])
                total7c += 1

        for i in range(len(topclasters)):
            if topclasters[i][0] in filled:
                continue
            if len(filled) == 5:
                break
            out.write(' ' + topclasters[i][0])
            filled.append(topclasters[i][0])
            total8 += 1

        out.write("\n")
    out.close()

    print ('')
    print ('Total 1: {} ...'.format(total1))
    print ('Total 2: {} ...'.format(total2))
    print ('Total 3: {} ...'.format(total3))
    print ('Total 4: {} ...'.format(total4))
    print ('Total 5: {} ...'.format(total5))

    print ('')
    print ('Total 6a: {} ...'.format(total6a))
    print ('Total 6b: {} ...'.format(total6b))
    print ('Total 6c: {} ...'.format(total6c))

    print ('')
    print ('Total 7a: {} ...'.format(total7a))
    print ('Total 7b: {} ...'.format(total7b))
    print ('Total 7c: {} ...'.format(total7c))

    print ('Total 8: {} ...'.format(total8))

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

best_s00,best_s01, best_s02, best_hotels_country, best_hotels_od_ulc, best_hotels_uid_miss, best_hotels_search_dest, popular_hotel_cluster, destination_clusters = prepare_arrays_match()
gen_submission(best_s00,best_s01, best_s02, best_hotels_country, best_hotels_od_ulc, best_hotels_uid_miss, best_hotels_search_dest, popular_hotel_cluster, destination_clusters)