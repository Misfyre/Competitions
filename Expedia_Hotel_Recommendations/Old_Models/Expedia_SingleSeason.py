from heapq import nlargest
from operator import itemgetter
from collections import defaultdict
import os

def month_to_season(month, day):

    if month == 12 and day >= 21: season = "Winter"
    elif month in (1,2): season = "Winter"
    elif month == 3 and day < 21: season = "Winter"
    elif month == 3 and day >= 21: season = "Spring"
    elif month in (4,5): season ="Spring"
    elif month == 6 and day < 21: season = "Spring"
    elif month == 6 and day >= 21: season = "Summer"
    elif month in (7,8): season = "Summer"
    elif month == 9 and day < 21: season = "Summer"
    elif month == 9 and day >= 21: season = "Fall"
    elif month in (10,11): season = "Fall"
    elif month == 12 and day < 21: season = "Fall"

    return season

def run_solution():

    print ('')
    print ('Preparing arrays...')
    f = open(os.path.expanduser('~/Desktop/Expedia/Data/train.csv'), "r")
    f.readline()

    best_hotels_od_ulc = defaultdict(lambda: defaultdict(int))
    best_hotels_search_dest = defaultdict(lambda: defaultdict(int))
    best_hotels_search_dest1 = defaultdict(lambda: defaultdict(int))
    best_hotel_country = defaultdict(lambda: defaultdict(int))

    spring_popular_hotel_cluster = defaultdict(int)
    summer_popular_hotel_cluster = defaultdict(int)
    fall_popular_hotel_cluster = defaultdict(int)
    winter_popular_hotel_cluster = defaultdict(int)

    popular_hotel_cluster = defaultdict(int)

    total = 0
    print ('')

    while 1:
        line = f.readline().strip()
        total += 1

        if total % 1000000 == 0:
            print('Reading {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")

        book_year = int(arr[0][:4])
        book_month = int(arr[0][5:7])
        book_day = int(arr[0][8:10])
        season = month_to_season(book_month, book_day)

        user_location_city = arr[5]
        orig_destination_distance = arr[6]
        srch_destination_id = arr[16]
        is_booking = int(arr[18])
        hotel_country = arr[21]
        hotel_market = arr[22]
        hotel_cluster = arr[23]

        append_0 = (book_year - 2012)*12 + book_month
        append_1 = 3 + 12*is_booking
        append_2 = 3 + 5*is_booking

        if season == "Spring":

            if user_location_city != '' and orig_destination_distance != '':
                hsh = (hash('Spring' + 'user_location_city_'+str(user_location_city) + '_orig_destination_distance_' + str(orig_destination_distance)))
                best_hotels_od_ulc[hsh][hotel_cluster] += append_0

            if srch_destination_id != '' and hotel_country != '' and hotel_market != '' and book_year == 2014:
                hsh = (hash('Spring' + 'srch_destination_id_' + str(srch_destination_id) + "_hotel_country_" + str(hotel_country) + "_hotel_market_"+str(hotel_market)))
                best_hotels_search_dest[hsh][hotel_cluster] += append_1

            if srch_destination_id != '':
                hsh = hash('Spring' + 'srch_destination_id_'+str(srch_destination_id))
                best_hotels_search_dest1[hsh][hotel_cluster] += append_1

            if hotel_country != '':
                hsh = hash('Spring' + 'hotel_country_'+str(hotel_country))
                best_hotel_country[hsh][hotel_cluster] += append_2

            spring_popular_hotel_cluster[hotel_cluster] += 1

        if season == "Summer":

            if user_location_city != '' and orig_destination_distance != '':
                hsh = (hash('Summer' + 'user_location_city_'+str(user_location_city) + '_orig_destination_distance_' + str(orig_destination_distance)))
                best_hotels_od_ulc[hsh][hotel_cluster] += append_0

            if srch_destination_id != '' and hotel_country != '' and hotel_market != '' and book_year == 2014:
                hsh = (hash('Summer' + 'srch_destination_id_' + str(srch_destination_id) + "_hotel_country_" + str(hotel_country) + "_hotel_market_"+str(hotel_market)))
                best_hotels_search_dest[hsh][hotel_cluster] += append_1

            if srch_destination_id != '':
                hsh = hash('Summer' + 'srch_destination_id_'+str(srch_destination_id))
                best_hotels_search_dest1[hsh][hotel_cluster] += append_1

            if hotel_country != '':
                hsh = hash('Summer' + 'hotel_country_'+str(hotel_country))
                best_hotel_country[hsh][hotel_cluster] += append_2

            summer_popular_hotel_cluster[hotel_cluster] += 1

        if season == "Fall":

            if user_location_city != '' and orig_destination_distance != '':
                hsh = (hash('Fall' + 'user_location_city_'+str(user_location_city) + '_orig_destination_distance_' + str(orig_destination_distance)))
                best_hotels_od_ulc[hsh][hotel_cluster] += append_0

            if srch_destination_id != '' and hotel_country != '' and hotel_market != '' and book_year == 2014:
                hsh = (hash('Fall' + 'srch_destination_id_' + str(srch_destination_id) + "_hotel_country_" + str(hotel_country) + "_hotel_market_"+str(hotel_market)))
                best_hotels_search_dest[hsh][hotel_cluster] += append_1

            if srch_destination_id != '':
                hsh = hash('Fall' + 'srch_destination_id_'+str(srch_destination_id))
                best_hotels_search_dest1[hsh][hotel_cluster] += append_1

            if hotel_country != '':
                hsh = hash('Fall' + 'hotel_country_'+str(hotel_country))
                best_hotel_country[hsh][hotel_cluster] += append_2

            fall_popular_hotel_cluster[hotel_cluster] += 1

        if season == "Winter":

            if user_location_city != '' and orig_destination_distance != '':
                hsh = (hash('Winter' + 'user_location_city_'+str(user_location_city) + '_orig_destination_distance_' + str(orig_destination_distance)))
                best_hotels_od_ulc[hsh][hotel_cluster] += append_0

            if srch_destination_id != '' and hotel_country != '' and hotel_market != '' and book_year == 2014:
                hsh = (hash('Winter' + 'srch_destination_id_' + str(srch_destination_id) + "_hotel_country_" + str(hotel_country) + "_hotel_market_"+str(hotel_market)))
                best_hotels_search_dest[hsh][hotel_cluster] += append_1

            if srch_destination_id != '':
                hsh = hash('Winter' + 'srch_destination_id_'+str(srch_destination_id))
                best_hotels_search_dest1[hsh][hotel_cluster] += append_1

            if hotel_country != '':
                hsh = hash('Winter' + 'hotel_country_'+str(hotel_country))
                best_hotel_country[hsh][hotel_cluster] += append_2

            winter_popular_hotel_cluster[hotel_cluster] += 1

        popular_hotel_cluster[hotel_cluster] += 1

    f.close()

    print ('')
    print ('Generating submission...')
    print ('')

    path = os.path.expanduser('~/Desktop/Expedia/Submission/submission.csv')

    out = open(path, "w")
    f = open(os.path.expanduser('~/Desktop/Expedia/Data/test.csv'), "r")
    f.readline()
    total = 0
    out.write("id,hotel_cluster\n")

    spring_topclusters = nlargest(5, sorted(spring_popular_hotel_cluster.items()), key=itemgetter(1))
    summer_topclusters = nlargest(5, sorted(summer_popular_hotel_cluster.items()), key=itemgetter(1))
    fall_topclusters = nlargest(5, sorted(fall_popular_hotel_cluster.items()), key=itemgetter(1))
    winter_topclusters = nlargest(5, sorted(winter_popular_hotel_cluster.items()), key=itemgetter(1))

    topclusters = nlargest(5, sorted(popular_hotel_cluster.items()), key=itemgetter(1))

    while 1:
        line = f.readline().strip()
        total += 1

        if total % 500000 == 0:
            print('Writing {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")

        id = arr[0]

        book_month = (int(arr[1][5:7]))
        book_day = int(arr[1][8:10])
        season = month_to_season(book_month, book_day)

        user_location_city = arr[6]
        orig_destination_distance = arr[7]
        srch_destination_id = arr[17]
        hotel_country = arr[20]
        hotel_market = arr[21]

        out.write(str(id) + ',')
        filled = []

        if season == "Spring":

            hsh = (hash('Spring' + 'user_location_city_'+str(user_location_city) + '_orig_destination_distance_' + str(orig_destination_distance)))

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

            hsh_1 = (hash('Spring' + 'srch_destination_id_' + str(srch_destination_id) + "_hotel_country_" + str(hotel_country) + "_hotel_market_"+str(hotel_market)))
            hsh_2 = (hash('Spring' + 'srch_destination_id_' + str(srch_destination_id)))

            if (len(filled) < 5) and (hsh_1 in best_hotels_search_dest):
                d = best_hotels_search_dest[hsh_1]
                topitems = nlargest(5, d.items(), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])

            elif (len(filled) < 5) and (hsh_2 in best_hotels_search_dest1):
                d = best_hotels_search_dest1[hsh_2]
                topitems = nlargest(5, d.items(), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])

            hsh = (hash('Spring' + 'hotel_country_' + str(hotel_country)))

            if (len(filled) < 5) and (hsh in best_hotel_country):
                d = best_hotel_country[hsh]
                topitems = nlargest(5, d.items(), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])

            if(len(filled) < 5):
                for i in range(len(spring_topclusters)):
                    if spring_topclusters[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + spring_topclusters[i][0])
                    filled.append(spring_topclusters[i][0])

        if season == "Summer":

            hsh = (hash('Summer' + 'user_location_city_'+str(user_location_city) + '_orig_destination_distance_' + str(orig_destination_distance)))

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

            hsh_1 = (hash('Summer' + 'srch_destination_id_' + str(srch_destination_id) + "_hotel_country_" + str(hotel_country) + "_hotel_market_"+str(hotel_market)))
            hsh_2 = (hash('Summer' + 'srch_destination_id_' + str(srch_destination_id)))

            if (len(filled) < 5) and (hsh_1 in best_hotels_search_dest):
                d = best_hotels_search_dest[hsh_1]
                topitems = nlargest(5, d.items(), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])

            elif (len(filled) < 5) and (hsh_2 in best_hotels_search_dest1):
                d = best_hotels_search_dest1[hsh_2]
                topitems = nlargest(5, d.items(), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])

            hsh = (hash('Summer' + 'hotel_country_' + str(hotel_country)))

            if (len(filled) < 5) and (hsh in best_hotel_country):
                d = best_hotel_country[hsh]
                topitems = nlargest(5, d.items(), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])

            if(len(filled) < 5):
                for i in range(len(summer_topclusters)):
                    if summer_topclusters[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + summer_topclusters[i][0])
                    filled.append(summer_topclusters[i][0])

        if season == "Fall":

            hsh = (hash('Fall' + 'user_location_city_'+str(user_location_city) + '_orig_destination_distance_' + str(orig_destination_distance)))

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

            hsh_1 = (hash('Fall' + 'srch_destination_id_' + str(srch_destination_id) + "_hotel_country_" + str(hotel_country) + "_hotel_market_"+str(hotel_market)))
            hsh_2 = (hash('Fall' + 'srch_destination_id_' + str(srch_destination_id)))

            if (len(filled) < 5) and (hsh_1 in best_hotels_search_dest):
                d = best_hotels_search_dest[hsh_1]
                topitems = nlargest(5, d.items(), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])

            elif (len(filled) < 5) and (hsh_2 in best_hotels_search_dest1):
                d = best_hotels_search_dest1[hsh_2]
                topitems = nlargest(5, d.items(), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])

            hsh = (hash('Fall' + 'hotel_country_' + str(hotel_country)))

            if (len(filled) < 5) and (hsh in best_hotel_country):
                d = best_hotel_country[hsh]
                topitems = nlargest(5, d.items(), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])

            if(len(filled) < 5):
                for i in range(len(fall_topclusters)):
                    if fall_topclusters[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + fall_topclusters[i][0])
                    filled.append(fall_topclusters[i][0])

        if season == "Winter":

            hsh = (hash('Winter' + 'user_location_city_'+str(user_location_city) + '_orig_destination_distance_' + str(orig_destination_distance)))

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

            hsh_1 = (hash('Winter' + 'srch_destination_id_' + str(srch_destination_id) + "_hotel_country_" + str(hotel_country) + "_hotel_market_"+str(hotel_market)))
            hsh_2 = (hash('Winter' + 'srch_destination_id_' + str(srch_destination_id)))

            if (len(filled) < 5) and (hsh_1 in best_hotels_search_dest):
                d = best_hotels_search_dest[hsh_1]
                topitems = nlargest(5, d.items(), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])

            elif (len(filled) < 5) and (hsh_2 in best_hotels_search_dest1):
                d = best_hotels_search_dest1[hsh_2]
                topitems = nlargest(5, d.items(), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])

            hsh = (hash('Winter' + 'hotel_country_' + str(hotel_country)))

            if (len(filled) < 5) and (hsh in best_hotel_country):
                d = best_hotel_country[hsh]
                topitems = nlargest(5, d.items(), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])

            if(len(filled) < 5):
                for i in range(len(winter_topclusters)):
                    if winter_topclusters[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + winter_topclusters[i][0])
                    filled.append(winter_topclusters[i][0])

        if(len(filled) < 5):
            for i in range(len(topclusters)):
                if topclusters[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topclusters[i][0])
                filled.append(topclusters[i][0])

        out.write("\n")
    out.close()

    print ('')
    print('Completed!')


if __name__ == "__main__":
    run_solution()
