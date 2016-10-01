import os
import pandas as pd
from heapq import nlargest
from operator import itemgetter
import ml_metrics as metrics
from datetime import date
import numpy as np
import math

def prepare_arrays_match():

    print ('')
    print ('Preparing Arrays...')
    print ('')

    f = open(os.path.expanduser('~/Desktop/Expedia/Data/train.csv'), "r")
    f.readline()

    best_hotels_od_ulc = dict()
    best_hotels_uid_miss = dict()
    best_hotels_search_dest = dict()
    best_hotels_country = dict()

    monthly_popular_hotel_cluster = dict()

    best_s00 = dict()
    best_s01 = dict()
    best_s02 = dict()

    total = 0

    while 1:

        line = f.readline().strip()
        total += 1

        if total % 8000000 == 0:
            print('Read {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")

        srch_ci = arr[11]
        srch_co = arr[12]

        if srch_ci != '':
            book_year = int(srch_ci[:4])
            book_month = int(srch_ci[5:7])
        else:
            book_year = int(arr[0][:4])
            book_month = int(arr[0][5:7])

        if srch_ci != '':
            srch_ci_month = int(srch_ci[5:7])
        else:
            srch_ci_month = int(arr[0][5:7])

        if srch_co != '':
            srch_co_month = int(srch_co[5:7])
        else:
            srch_co_month = ''

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

        if hotel_market != '':
            s3 = (hotel_market)
            if s3 in best_hotels_country:
                if hotel_cluster in best_hotels_country[s3]:
                    best_hotels_country[s3][hotel_cluster] += append_2
                else:
                    best_hotels_country[s3][hotel_cluster] = append_2
            else:
                best_hotels_country[s3] = dict()
                best_hotels_country[s3][hotel_cluster] = append_2

        s4 = (srch_ci_month)
        if s4 in monthly_popular_hotel_cluster:
            if hotel_cluster in monthly_popular_hotel_cluster:
                monthly_popular_hotel_cluster[s4][hotel_cluster] += append_0
            else:
                monthly_popular_hotel_cluster[s4][hotel_cluster] = append_0
        else:
            monthly_popular_hotel_cluster[s4] = dict()
            monthly_popular_hotel_cluster[s4][hotel_cluster] = append_0

    f.close()
    return best_s00,best_s01, best_s02, best_hotels_country, best_hotels_od_ulc, best_hotels_uid_miss, best_hotels_search_dest, monthly_popular_hotel_cluster


def gen_submission(best_s00, best_s01,best_s02, best_hotels_country, best_hotels_search_dest, best_hotels_od_ulc, best_hotels_uid_miss, monthly_popular_hotel_cluster):

    print ('')
    path = os.path.expanduser('~/Desktop/Expedia/Submission/submission.csv')
    out = open(path, "w")
    f = open(os.path.expanduser('~/Desktop/Expedia/Data/test.csv'), "r")
    f.readline()

    total = 0
    total0 = 0
    total00 = 0
    total000 = 0
    total1 = 0
    total2 = 0
    total3 = 0
    total4 = 0

    out.write("id,hotel_cluster\n")

    while 1:

        line = f.readline().strip()
        total += 1

        if total % 500000 == 0:
            print('Write {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")

        srch_ci = arr[12]
        srch_co = arr[13]

        if srch_ci != '':
            srch_ci_month = int(srch_ci[5:7])
        else:
            srch_ci_month = ''

        if srch_co != '':
            srch_co_month = int(srch_co[5:7])
        else:
            srch_co_month = ''

        id = arr[0]
        user_location_city = arr[6]
        orig_destination_distance = arr[7]
        user_id = arr[8]
        srch_destination_id = arr[17]
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
            s0 = (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
            if s0 in best_hotels_uid_miss:
                d = best_hotels_uid_miss[s0]
                topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])
                    total0 += 1

        s00 = (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
        s01 = (user_id, srch_destination_id, hotel_country, hotel_market)
        if s01 in best_s01 and s00 not in best_s00:
            d = best_s01[s01]
            topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                total00 += 1

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
                total000 += 1

        s2 = (srch_destination_id,hotel_country,hotel_market)
        if s2 in best_hotels_search_dest:
            d = best_hotels_search_dest[s2]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                total2 += 1

        s3 = (hotel_market)
        if s3 in best_hotels_country:
            d = best_hotels_country[s3]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                total3 += 1

        s4 = (srch_ci_month)
        if s4 in monthly_popular_hotel_cluster:
            d = monthly_popular_hotel_cluster[s4]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                total4 += 1

        out.write("\n")
    out.close()

    print ('')
    print ('Total 0: {} ...'.format(total0))
    print ('Total 00: {} ...'.format(total00))
    print ('Total 000: {} ...'.format(total000))
    print ('Total 1: {} ...'.format(total1))
    print ('Total 2: {} ...'.format(total2))
    print ('Total 3: {} ...'.format(total3))
    print ('Total 4: {} ...'.format(total4))

    print ('')
    print ('Completed!..')

best_s00, best_s01, best_s02, best_hotels_country, best_hotels_od_ulc, best_hotels_uid_miss, best_hotels_search_dest, popular_hotel_cluster, monthly_popular_hotel_cluster = prepare_arrays_match()
gen_submission(best_s00, best_s01, best_s02, best_hotels_country, best_hotels_search_dest, best_hotels_od_ulc, best_hotels_uid_miss, popular_hotel_cluster, monthly_popular_hotel_cluster)
