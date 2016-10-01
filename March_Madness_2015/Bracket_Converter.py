__author__ = 'Nick Sarris'

import numpy as np
import pandas as pd

Bracket_Teams = pd.read_csv('~/Desktop/Massey Predictor/Bracket_Data/Bracket Teams.csv', encoding = "ISO-8859-1")
Bracket_Seeds = pd.read_csv('~/Desktop/Massey Predictor/Bracket_Data/Bracket Seeds.csv', encoding = "ISO-8859-1")

Predictions = pd.read_csv('~/Desktop/Massey Predictor/Kaggle_MM_Data/predictions/monte-mcnair_293071_2711520.csv', encoding = "ISO-8859-1")

def Split_Array(Input, Season):

    Input_Array = np.asarray(Input)

    list_1 = []
    for a in Input_Array:
        if a[0] == Season:
            list.append(list_1, a)

    return list_1

def Split_Bracket(Input, Region):

    Input_Array = np.asarray(Input)

    list_1 = []
    for a in Input_Array:
        if a[1][0] == Region:
            list.append(list_1, a)

    return list_1

def East_Bracket(Bracket_East_Seeds, Predictions, Team_Dictionary):

    print ('EAST RESULTS')

    print ('-------------------------')
    print ('-------------------------')
    try:
        PreMatch_1 = ('2016_%d_%d' % (Bracket_East_Seeds['W11a'], Bracket_East_Seeds['W11b']))

        if Predictions[PreMatch_1] >= .5:
            Winner_PreMatch_1 = Bracket_East_Seeds['W11a']
        elif Predictions[PreMatch_1] <= .5:
            Winner_PreMatch_1 = Bracket_East_Seeds['W11b']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_East_Seeds['W11a']], (100 * Predictions[PreMatch_1]), Team_Dictionary[Bracket_East_Seeds['W11b']], (100 * (1 - Predictions[PreMatch_1]))), 'Winner =', Team_Dictionary[Winner_PreMatch_1])

    except:
        PreMatch_1 = ('2016_%d_%d' % (Bracket_East_Seeds['W11b'], Bracket_East_Seeds['W11a']))

        if Predictions[PreMatch_1] >= .5:
            Winner_PreMatch_1 = Bracket_East_Seeds['W11b']
        elif Predictions[PreMatch_1] <= .5:
            Winner_PreMatch_1 = Bracket_East_Seeds['W11a']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_East_Seeds['W11b']], (100 * Predictions[PreMatch_1]), Team_Dictionary[Bracket_East_Seeds['W11a']], (100 * (1 - Predictions[PreMatch_1]))), 'Winner =', Team_Dictionary[Winner_PreMatch_1])

    try:
        PreMatch_2 = ('2016_%d_%d' % (Bracket_East_Seeds['W16a'], Bracket_East_Seeds['W16b']))

        if Predictions[PreMatch_2] >= .5:
            Winner_PreMatch_2 = Bracket_East_Seeds['W16a']
        elif Predictions[PreMatch_2] <= .5:
            Winner_PreMatch_2 = Bracket_East_Seeds['W16b']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_East_Seeds['W16a']], (100 * Predictions[PreMatch_2]), Team_Dictionary[Bracket_East_Seeds['W16b']], (100 * (1 - Predictions[PreMatch_2]))), 'Winner =', Team_Dictionary[Winner_PreMatch_2])

    except:
        PreMatch_2 = ('2016_%d_%d' % (Bracket_East_Seeds['W16b'], Bracket_East_Seeds['W16a']))


        if Predictions[PreMatch_2] >= .5:
            Winner_PreMatch_2 = Bracket_East_Seeds['W16b']
        elif Predictions[PreMatch_2] <= .5:
            Winner_PreMatch_2 = Bracket_East_Seeds['W16a']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_East_Seeds['W16b']], (100 * Predictions[PreMatch_2]), Team_Dictionary[Bracket_East_Seeds['W16a']], (100 * (1 - Predictions[PreMatch_2]))), 'Winner =', Team_Dictionary[Winner_PreMatch_2])

    print ('-------------------------')

    try:
        Match_1 = ('2016_%d_%d' % (Winner_PreMatch_2, Bracket_East_Seeds['W01']))

        if Predictions[Match_1] >= .5:
            Winner_Match_1 = Winner_PreMatch_2
        elif Predictions[Match_1] <= .5:
            Winner_Match_1 = Bracket_East_Seeds['W01']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_PreMatch_2], (100 * (Predictions[Match_1])), Team_Dictionary[Bracket_East_Seeds['W01']], (100 * (1 - Predictions[Match_1]))), 'Winner =', Team_Dictionary[Winner_Match_1])

    except:
        Match_1 = ('2016_%d_%d' % (Bracket_East_Seeds['W01'], Winner_PreMatch_2))

        if Predictions[Match_1] >= .5:
            Winner_Match_1 = Bracket_East_Seeds['W01']
        elif Predictions[Match_1] <= .5:
            Winner_Match_1 = Winner_PreMatch_2

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_East_Seeds['W01']], (100 * (Predictions[Match_1])), Team_Dictionary[Winner_PreMatch_2], (100 * (1 - Predictions[Match_1]))), 'Winner =', Team_Dictionary[Winner_Match_1])

    try:
        Match_2 = ('2016_%d_%d' % (Bracket_East_Seeds['W09'], Bracket_East_Seeds['W08']))

        if Predictions[Match_2] >= .5:
            Winner_Match_2 = Bracket_East_Seeds['W09']
        elif Predictions[Match_2] <= .5:
            Winner_Match_2 = Bracket_East_Seeds['W08']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_East_Seeds['W09']], (100 * Predictions[Match_2]), Team_Dictionary[Bracket_East_Seeds['W08']], (100 * (1 - Predictions[Match_2]))), 'Winner =', Team_Dictionary[Winner_Match_2])

    except:
        Match_2 = ('2016_%d_%d' % (Bracket_East_Seeds['W08'], Bracket_East_Seeds['W09']))

        if Predictions[Match_2] >= .5:
            Winner_Match_2 = Bracket_East_Seeds['W08']
        elif Predictions[Match_2] <= .5:
            Winner_Match_2 = Bracket_East_Seeds['W09']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_East_Seeds['W08']], (100 * Predictions[Match_2]), Team_Dictionary[Bracket_East_Seeds['W09']], (100 * (1 - Predictions[Match_2]))), 'Winner =', Team_Dictionary[Winner_Match_2])

    try:
        Match_3 = ('2016_%d_%d' % (Bracket_East_Seeds['W12'], Bracket_East_Seeds['W05']))

        if Predictions[Match_3] >= .5:
            Winner_Match_3 = Bracket_East_Seeds['W12']
        elif Predictions[Match_3] <= .5:
            Winner_Match_3 = Bracket_East_Seeds['W05']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_East_Seeds['W12']], (100 * Predictions[Match_3]), Team_Dictionary[Bracket_East_Seeds['W05']], (100 * (1 - Predictions[Match_3]))), 'Winner =', Team_Dictionary[Winner_Match_3])

    except:
        Match_3 = ('2016_%d_%d' % (Bracket_East_Seeds['W05'], Bracket_East_Seeds['W12']))

        if Predictions[Match_3] >= .5:
            Winner_Match_3 = Bracket_East_Seeds['W05']
        elif Predictions[Match_3] <= .5:
            Winner_Match_3 = Bracket_East_Seeds['W12']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_East_Seeds['W05']], (100 * Predictions[Match_3]), Team_Dictionary[Bracket_East_Seeds['W12']], (100 * (1 - Predictions[Match_3]))), 'Winner =', Team_Dictionary[Winner_Match_3])

    try:
        Match_4 = ('2016_%d_%d' % (Bracket_East_Seeds['W04'], Bracket_East_Seeds['W13']))

        if Predictions[Match_4] >= .5:
            Winner_Match_4 = Bracket_East_Seeds['W04']
        elif Predictions[Match_4] <= .5:
            Winner_Match_4 = Bracket_East_Seeds['W13']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_East_Seeds['W04']], (100 * Predictions[Match_4]), Team_Dictionary[Bracket_East_Seeds['W13']], (100 * (1 - Predictions[Match_4]))), 'Winner =', Team_Dictionary[Winner_Match_4])

    except:
        Match_4 = ('2016_%d_%d' % (Bracket_East_Seeds['W13'], Bracket_East_Seeds['W04']))

        if Predictions[Match_4] >= .5:
            Winner_Match_4 = Bracket_East_Seeds['W13']
        elif Predictions[Match_4] <= .5:
            Winner_Match_4 = Bracket_East_Seeds['W04']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_East_Seeds['W13']], (100 * Predictions[Match_4]), Team_Dictionary[Bracket_East_Seeds['W04']], (100 * (1 - Predictions[Match_4]))), 'Winner =', Team_Dictionary[Winner_Match_4])

    try:
        Match_5 = ('2016_%d_%d' % (Winner_PreMatch_1, Bracket_East_Seeds['W06']))

        if Predictions[Match_5] >= .5:
            Winner_Match_5 = Winner_PreMatch_1
        elif Predictions[Match_5] <= .5:
            Winner_Match_5 = Bracket_East_Seeds['W06']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_PreMatch_1], (100 * Predictions[Match_5]), Team_Dictionary[Bracket_East_Seeds['W06']], (100 * (1 - Predictions[Match_5]))), 'Winner =', Team_Dictionary[Winner_Match_5])

    except:
        Match_5 = ('2016_%d_%d' % (Bracket_East_Seeds['W06'], Winner_PreMatch_1))

        if Predictions[Match_5] >= .5:
            Winner_Match_5 = Bracket_East_Seeds['W06']
        elif Predictions[Match_5] <= .5:
            Winner_Match_5 = Winner_PreMatch_1

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_East_Seeds['W06']], (100 * Predictions[Match_5]), Team_Dictionary[Winner_PreMatch_1], (100 * (1 - Predictions[Match_5]))), 'Winner =', Team_Dictionary[Winner_Match_5])

    try:
        Match_6 = ('2016_%d_%d' % (Bracket_East_Seeds['W14'], Bracket_East_Seeds['W03']))

        if Predictions[Match_6] >= .5:
            Winner_Match_6 = Bracket_East_Seeds['W14']
        elif Predictions[Match_6] <= .5:
            Winner_Match_6 = Bracket_East_Seeds['W03']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_East_Seeds['W14']], (100 * Predictions[Match_6]), Team_Dictionary[Bracket_East_Seeds['W03']], (100 * (1 - Predictions[Match_6]))), 'Winner =', Team_Dictionary[Winner_Match_6])

    except:
        Match_6 = ('2016_%d_%d' % (Bracket_East_Seeds['W03'], Bracket_East_Seeds['W14']))

        if Predictions[Match_6] >= .5:
            Winner_Match_6 = Bracket_East_Seeds['W03']
        elif Predictions[Match_6] <= .5:
            Winner_Match_6 = Bracket_East_Seeds['W14']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_East_Seeds['W03']], (100 * Predictions[Match_6]), Team_Dictionary[Bracket_East_Seeds['W14']], (100 * (1 - Predictions[Match_6]))), 'Winner =', Team_Dictionary[Winner_Match_6])

    try:
        Match_7 = ('2016_%d_%d' % (Bracket_East_Seeds['W07'], Bracket_East_Seeds['W10']))

        if Predictions[Match_7] >= .5:
            Winner_Match_7 = Bracket_East_Seeds['W07']
        elif Predictions[Match_7] <= .5:
            Winner_Match_7 = Bracket_East_Seeds['W10']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_East_Seeds['W07']], (100 * Predictions[Match_7]), Team_Dictionary[Bracket_East_Seeds['W10']], (100 * (1 - Predictions[Match_7]))), 'Winner =', Team_Dictionary[Winner_Match_7])

    except:
        Match_7 = ('2016_%d_%d' % (Bracket_East_Seeds['W10'], Bracket_East_Seeds['W07']))

        if Predictions[Match_7] >= .5:
            Winner_Match_7 = Bracket_East_Seeds['W10']
        elif Predictions[Match_7] <= .5:
            Winner_Match_7 = Bracket_East_Seeds['W07']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_East_Seeds['W10']], (100 * Predictions[Match_7]), Team_Dictionary[Bracket_East_Seeds['W07']], (100 * (1 - Predictions[Match_7]))), 'Winner =', Team_Dictionary[Winner_Match_7])

    try:
        Match_8 = ('2016_%d_%d' % (Bracket_East_Seeds['W15'], Bracket_East_Seeds['W02']))

        if Predictions[Match_8] >= .5:
            Winner_Match_8 = Bracket_East_Seeds['W15']
        elif Predictions[Match_8] <= .5:
            Winner_Match_8 = Bracket_East_Seeds['W02']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_East_Seeds['W15']], (100 * Predictions[Match_8]), Team_Dictionary[Bracket_East_Seeds['W02']], (100 * (1 - Predictions[Match_8]))), 'Winner =', Team_Dictionary[Winner_Match_8])

    except:
        Match_8 = ('2016_%d_%d' % (Bracket_East_Seeds['W02'], Bracket_East_Seeds['W15']))

        if Predictions[Match_8] >= .5:
            Winner_Match_8 = Bracket_East_Seeds['W02']
        elif Predictions[Match_8] <= .5:
            Winner_Match_8 = Bracket_East_Seeds['W15']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_East_Seeds['W02']], (100 * Predictions[Match_8]), Team_Dictionary[Bracket_East_Seeds['W15']], (100 * (1 - Predictions[Match_8]))), 'Winner =', Team_Dictionary[Winner_Match_8])

    print ('-------------------------')

    try:
        Match_9 = ('2016_%d_%d' % (Winner_Match_1, Winner_Match_2))

        if Predictions[Match_9] >= .5:
            Winner_Match_9 = Winner_Match_1
        elif Predictions[Match_9] <= .5:
            Winner_Match_9 = Winner_Match_2

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_1], (100 * Predictions[Match_9]), Team_Dictionary[Winner_Match_2], (100 * (1 - Predictions[Match_9]))), 'Winner =', Team_Dictionary[Winner_Match_9])

    except:
        Match_9 = ('2016_%d_%d' % (Winner_Match_2, Winner_Match_1))

        if Predictions[Match_9] >= .5:
            Winner_Match_9 = Winner_Match_2
        elif Predictions[Match_9] <= .5:
            Winner_Match_9 = Winner_Match_1

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_2], (100 * Predictions[Match_9]), Team_Dictionary[Winner_Match_1], (100 * (1 - Predictions[Match_9]))), 'Winner =', Team_Dictionary[Winner_Match_9])

    try:
        Match_10 = ('2016_%d_%d' % (Winner_Match_3, Winner_Match_4))

        if Predictions[Match_10] >= .5:
            Winner_Match_10 = Winner_Match_3
        elif Predictions[Match_10] <= .5:
            Winner_Match_10 = Winner_Match_4

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_3], (100 * Predictions[Match_10]), Team_Dictionary[Winner_Match_4], (100 * (1 - Predictions[Match_10]))), 'Winner =', Team_Dictionary[Winner_Match_10])

    except:
        Match_10 = ('2016_%d_%d' % (Winner_Match_4, Winner_Match_3))

        if Predictions[Match_10] >= .5:
            Winner_Match_10 = Winner_Match_4
        elif Predictions[Match_10] <= .5:
            Winner_Match_10 = Winner_Match_3

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_4], (100 * Predictions[Match_10]), Team_Dictionary[Winner_Match_3], (100 * (1 - Predictions[Match_10]))), 'Winner =', Team_Dictionary[Winner_Match_10])

    try:
        Match_11 = ('2016_%d_%d' % (Winner_Match_5, Winner_Match_6))

        if Predictions[Match_11] >= .5:
            Winner_Match_11 = Winner_Match_5
        elif Predictions[Match_11] <= .5:
            Winner_Match_11 = Winner_Match_6

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_5], (100 * Predictions[Match_11]), Team_Dictionary[Winner_Match_6], (100 * (1 - Predictions[Match_11]))), 'Winner =', Team_Dictionary[Winner_Match_11])

    except:
        Match_11 = ('2016_%d_%d' % (Winner_Match_6, Winner_Match_5))

        if Predictions[Match_11] >= .5:
            Winner_Match_11 = Winner_Match_6
        elif Predictions[Match_11] <= .5:
            Winner_Match_11 = Winner_Match_5

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_6], (100 * Predictions[Match_11]), Team_Dictionary[Winner_Match_5], (100 * (1 - Predictions[Match_11]))), 'Winner =', Team_Dictionary[Winner_Match_11])

    try:
        Match_12 = ('2016_%d_%d' % (Winner_Match_7, Winner_Match_8))

        if Predictions[Match_12] >= .5:
            Winner_Match_12 = Winner_Match_7
        elif Predictions[Match_12] <= .5:
            Winner_Match_12 = Winner_Match_8

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_7], (100 * Predictions[Match_12]), Team_Dictionary[Winner_Match_8], (100 * (1 - Predictions[Match_12]))), 'Winner =', Team_Dictionary[Winner_Match_12])

    except:
        Match_12 = ('2016_%d_%d' % (Winner_Match_8, Winner_Match_7))

        if Predictions[Match_12] >= .5:
            Winner_Match_12 = Winner_Match_8
        elif Predictions[Match_12] <= .5:
            Winner_Match_12 = Winner_Match_7

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_8], (100 * Predictions[Match_12]), Team_Dictionary[Winner_Match_7], (100 * (1 - Predictions[Match_12]))), 'Winner =', Team_Dictionary[Winner_Match_12])

    print ('-------------------------')

    try:
        Match_13 = ('2016_%d_%d' % (Winner_Match_10, Winner_Match_9))

        if Predictions[Match_13] >= .5:
            Winner_Match_13 = Winner_Match_10
        elif Predictions[Match_13] <= .5:
            Winner_Match_13 = Winner_Match_9

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_10], (100 * Predictions[Match_13]), Team_Dictionary[Winner_Match_9], (100 * (1 - Predictions[Match_13]))), 'Winner =', Team_Dictionary[Winner_Match_13])

    except:
        Match_13 = ('2016_%d_%d' % (Winner_Match_9, Winner_Match_10))

        if Predictions[Match_13] >= .5:
            Winner_Match_13 = Winner_Match_9
        elif Predictions[Match_13] <= .5:
            Winner_Match_13 = Winner_Match_10

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_9], (100 * Predictions[Match_13]), Team_Dictionary[Winner_Match_10], (100 * (1 - Predictions[Match_13]))), 'Winner =', Team_Dictionary[Winner_Match_13])

    try:
        Match_14 = ('2016_%d_%d' % (Winner_Match_11, Winner_Match_12))

        if Predictions[Match_14] >= .5:
            Winner_Match_14 = Winner_Match_11
        elif Predictions[Match_14] <= .5:
            Winner_Match_14 = Winner_Match_12

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_11], (100 * Predictions[Match_14]), Team_Dictionary[Winner_Match_12], (100 * (1 - Predictions[Match_14]))), 'Winner =', Team_Dictionary[Winner_Match_14])

    except:
        Match_14 = ('2016_%d_%d' % (Winner_Match_12, Winner_Match_11))

        if Predictions[Match_14] >= .5:
            Winner_Match_14 = Winner_Match_12
        elif Predictions[Match_14] <= .5:
            Winner_Match_14 = Winner_Match_11

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_12], (100 * Predictions[Match_14]), Team_Dictionary[Winner_Match_11], (100 * (1 - Predictions[Match_14]))), 'Winner =', Team_Dictionary[Winner_Match_14])

    print ('-------------------------')

    try:
        Match_15 = ('2016_%d_%d' % (Winner_Match_13, Winner_Match_14))

        if Predictions[Match_15] >= .5:
            Winner_Match_15 = Winner_Match_13
        elif Predictions[Match_15] <= .5:
            Winner_Match_15 = Winner_Match_14

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_13], (100 * Predictions[Match_15]), Team_Dictionary[Winner_Match_14], (100 * (1 - Predictions[Match_15]))), 'Winner =', Team_Dictionary[Winner_Match_15])

    except:
        Match_15 = ('2016_%d_%d' % (Winner_Match_14, Winner_Match_13))

        if Predictions[Match_15] >= .5:
            Winner_Match_15 = Winner_Match_14
        elif Predictions[Match_15] <= .5:
            Winner_Match_15 = Winner_Match_13

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_14], (100 * Predictions[Match_15]), Team_Dictionary[Winner_Match_13], (100 * (1 - Predictions[Match_15]))), 'Winner =', Team_Dictionary[Winner_Match_15])

    return (Winner_Match_15)

def Midwest_Bracket(Bracket_Midwest_Seeds, Predictions, Team_Dictionary):

    print ('')
    print ('')

    print ('MIDWEST RESULTS')

    print ('-------------------------')
    print ('-------------------------')

    try:
        Match_1 = ('2016_%d_%d' % (Bracket_Midwest_Seeds['X16'], Bracket_Midwest_Seeds['X01']))

        if Predictions[Match_1] >= .5:
            Winner_Match_1 = Bracket_Midwest_Seeds['X16']
        elif Predictions[Match_1] <= .5:
            Winner_Match_1 = Bracket_Midwest_Seeds['X01']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_Midwest_Seeds['X16']], (100 * (Predictions[Match_1])), Team_Dictionary[Bracket_Midwest_Seeds['X01']], (100 * (1 - Predictions[Match_1]))), 'Winner =', Team_Dictionary[Winner_Match_1])

    except:
        Match_1 = ('2016_%d_%d' % (Bracket_Midwest_Seeds['X01'], Bracket_Midwest_Seeds['X16']))

        if Predictions[Match_1] >= .5:
            Winner_Match_1 = Bracket_Midwest_Seeds['X01']
        elif Predictions[Match_1] <= .5:
            Winner_Match_1 = Bracket_Midwest_Seeds['X16']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_Midwest_Seeds['X01']], (100 * (Predictions[Match_1])), Team_Dictionary[Bracket_Midwest_Seeds['X16']], (100 * (1 - Predictions[Match_1]))), 'Winner =', Team_Dictionary[Winner_Match_1])

    try:
        Match_2 = ('2016_%d_%d' % (Bracket_Midwest_Seeds['X09'], Bracket_Midwest_Seeds['X08']))

        if Predictions[Match_2] >= .5:
            Winner_Match_2 = Bracket_Midwest_Seeds['X09']
        elif Predictions[Match_2] <= .5:
            Winner_Match_2 = Bracket_Midwest_Seeds['X08']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_Midwest_Seeds['X09']], (100 * Predictions[Match_2]), Team_Dictionary[Bracket_Midwest_Seeds['X08']], (100 * (1 - Predictions[Match_2]))), 'Winner =', Team_Dictionary[Winner_Match_2])

    except:
        Match_2 = ('2016_%d_%d' % (Bracket_Midwest_Seeds['X08'], Bracket_Midwest_Seeds['X09']))

        if Predictions[Match_2] >= .5:
            Winner_Match_2 = Bracket_Midwest_Seeds['X08']
        elif Predictions[Match_2] <= .5:
            Winner_Match_2 = Bracket_Midwest_Seeds['X09']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_Midwest_Seeds['X08']], (100 * Predictions[Match_2]), Team_Dictionary[Bracket_Midwest_Seeds['X09']], (100 * (1 - Predictions[Match_2]))), 'Winner =', Team_Dictionary[Winner_Match_2])

    try:
        Match_3 = ('2016_%d_%d' % (Bracket_Midwest_Seeds['X12'], Bracket_Midwest_Seeds['X05']))

        if Predictions[Match_3] >= .5:
            Winner_Match_3 = Bracket_Midwest_Seeds['X12']
        elif Predictions[Match_3] <= .5:
            Winner_Match_3 = Bracket_Midwest_Seeds['X05']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_Midwest_Seeds['X12']], (100 * Predictions[Match_3]), Team_Dictionary[Bracket_Midwest_Seeds['X05']], (100 * (1 - Predictions[Match_3]))), 'Winner =', Team_Dictionary[Winner_Match_3])

    except:
        Match_3 = ('2016_%d_%d' % (Bracket_Midwest_Seeds['X05'], Bracket_Midwest_Seeds['X12']))

        if Predictions[Match_3] >= .5:
            Winner_Match_3 = Bracket_Midwest_Seeds['X05']
        elif Predictions[Match_3] <= .5:
            Winner_Match_3 = Bracket_Midwest_Seeds['X12']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_Midwest_Seeds['X05']], (100 * Predictions[Match_3]), Team_Dictionary[Bracket_Midwest_Seeds['X12']], (100 * (1 - Predictions[Match_3]))), 'Winner =', Team_Dictionary[Winner_Match_3])

    try:
        Match_4 = ('2016_%d_%d' % (Bracket_Midwest_Seeds['X04'], Bracket_Midwest_Seeds['X13']))

        if Predictions[Match_4] >= .5:
            Winner_Match_4 = Bracket_Midwest_Seeds['X04']
        elif Predictions[Match_4] <= .5:
            Winner_Match_4 = Bracket_Midwest_Seeds['X13']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_Midwest_Seeds['X04']], (100 * Predictions[Match_4]), Team_Dictionary[Bracket_Midwest_Seeds['X13']], (100 * (1 - Predictions[Match_4]))), 'Winner =', Team_Dictionary[Winner_Match_4])

    except:
        Match_4 = ('2016_%d_%d' % (Bracket_Midwest_Seeds['X13'], Bracket_Midwest_Seeds['X04']))

        if Predictions[Match_4] >= .5:
            Winner_Match_4 = Bracket_Midwest_Seeds['X13']
        elif Predictions[Match_4] <= .5:
            Winner_Match_4 = Bracket_Midwest_Seeds['X04']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_Midwest_Seeds['X13']], (100 * Predictions[Match_4]), Team_Dictionary[Bracket_Midwest_Seeds['X04']], (100 * (1 - Predictions[Match_4]))), 'Winner =', Team_Dictionary[Winner_Match_4])

    try:
        Match_5 = ('2016_%d_%d' % (Bracket_Midwest_Seeds['X11'], Bracket_Midwest_Seeds['X06']))

        if Predictions[Match_5] >= .5:
            Winner_Match_5 = Bracket_Midwest_Seeds['X11']
        elif Predictions[Match_5] <= .5:
            Winner_Match_5 = Bracket_Midwest_Seeds['X06']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_Midwest_Seeds['X11']], (100 * Predictions[Match_5]), Team_Dictionary[Bracket_Midwest_Seeds['X06']], (100 * (1 - Predictions[Match_5]))), 'Winner =', Team_Dictionary[Winner_Match_5])

    except:
        Match_5 = ('2016_%d_%d' % (Bracket_Midwest_Seeds['X06'], Bracket_Midwest_Seeds['X11']))

        if Predictions[Match_5] >= .5:
            Winner_Match_5 = Bracket_Midwest_Seeds['X06']
        elif Predictions[Match_5] <= .5:
            Winner_Match_5 = Bracket_Midwest_Seeds['X11']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_Midwest_Seeds['X06']], (100 * Predictions[Match_5]), Team_Dictionary[Bracket_Midwest_Seeds['X11']], (100 * (1 - Predictions[Match_5]))), 'Winner =', Team_Dictionary[Winner_Match_5])

    try:
        Match_6 = ('2016_%d_%d' % (Bracket_Midwest_Seeds['X14'], Bracket_Midwest_Seeds['X03']))

        if Predictions[Match_6] >= .5:
            Winner_Match_6 = Bracket_Midwest_Seeds['X14']
        elif Predictions[Match_6] <= .5:
            Winner_Match_6 = Bracket_Midwest_Seeds['X03']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_Midwest_Seeds['X14']], (100 * Predictions[Match_6]), Team_Dictionary[Bracket_Midwest_Seeds['X03']], (100 * (1 - Predictions[Match_6]))), 'Winner =', Team_Dictionary[Winner_Match_6])

    except:
        Match_6 = ('2016_%d_%d' % (Bracket_Midwest_Seeds['X03'], Bracket_Midwest_Seeds['X14']))

        if Predictions[Match_6] >= .5:
            Winner_Match_6 = Bracket_Midwest_Seeds['X03']
        elif Predictions[Match_6] <= .5:
            Winner_Match_6 = Bracket_Midwest_Seeds['X14']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_Midwest_Seeds['X03']], (100 * Predictions[Match_6]), Team_Dictionary[Bracket_Midwest_Seeds['X14']], (100 * (1 - Predictions[Match_6]))), 'Winner =', Team_Dictionary[Winner_Match_6])

    try:
        Match_7 = ('2016_%d_%d' % (Bracket_Midwest_Seeds['X07'], Bracket_Midwest_Seeds['X10']))

        if Predictions[Match_7] >= .5:
            Winner_Match_7 = Bracket_Midwest_Seeds['X07']
        elif Predictions[Match_7] <= .5:
            Winner_Match_7 = Bracket_Midwest_Seeds['X10']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_Midwest_Seeds['X07']], (100 * Predictions[Match_7]), Team_Dictionary[Bracket_Midwest_Seeds['X10']], (100 * (1 - Predictions[Match_7]))), 'Winner =', Team_Dictionary[Winner_Match_7])

    except:
        Match_7 = ('2016_%d_%d' % (Bracket_Midwest_Seeds['X10'], Bracket_Midwest_Seeds['X07']))

        if Predictions[Match_7] >= .5:
            Winner_Match_7 = Bracket_Midwest_Seeds['X10']
        elif Predictions[Match_7] <= .5:
            Winner_Match_7 = Bracket_Midwest_Seeds['X07']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_Midwest_Seeds['X10']], (100 * Predictions[Match_7]), Team_Dictionary[Bracket_Midwest_Seeds['X07']], (100 * (1 - Predictions[Match_7]))), 'Winner =', Team_Dictionary[Winner_Match_7])

    try:
        Match_8 = ('2016_%d_%d' % (Bracket_Midwest_Seeds['X15'], Bracket_Midwest_Seeds['X02']))

        if Predictions[Match_8] >= .5:
            Winner_Match_8 = Bracket_Midwest_Seeds['X15']
        elif Predictions[Match_8] <= .5:
            Winner_Match_8 = Bracket_Midwest_Seeds['X02']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_Midwest_Seeds['X15']], (100 * Predictions[Match_8]), Team_Dictionary[Bracket_Midwest_Seeds['X02']], (100 * (1 - Predictions[Match_8]))), 'Winner =', Team_Dictionary[Winner_Match_8])

    except:
        Match_8 = ('2016_%d_%d' % (Bracket_Midwest_Seeds['X02'], Bracket_Midwest_Seeds['X15']))

        if Predictions[Match_8] >= .5:
            Winner_Match_8 = Bracket_Midwest_Seeds['X02']
        elif Predictions[Match_8] <= .5:
            Winner_Match_8 = Bracket_Midwest_Seeds['X15']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_Midwest_Seeds['X02']], (100 * Predictions[Match_8]), Team_Dictionary[Bracket_Midwest_Seeds['X15']], (100 * (1 - Predictions[Match_8]))), 'Winner =', Team_Dictionary[Winner_Match_8])

    print ('-------------------------')

    try:
        Match_9 = ('2016_%d_%d' % (Winner_Match_1, Winner_Match_2))

        if Predictions[Match_9] >= .5:
            Winner_Match_9 = Winner_Match_1
        elif Predictions[Match_9] <= .5:
            Winner_Match_9 = Winner_Match_2

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_1], (100 * Predictions[Match_9]), Team_Dictionary[Winner_Match_2], (100 * (1 - Predictions[Match_9]))), 'Winner =', Team_Dictionary[Winner_Match_9])

    except:
        Match_9 = ('2016_%d_%d' % (Winner_Match_2, Winner_Match_1))

        if Predictions[Match_9] >= .5:
            Winner_Match_9 = Winner_Match_2
        elif Predictions[Match_9] <= .5:
            Winner_Match_9 = Winner_Match_1

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_2], (100 * Predictions[Match_9]), Team_Dictionary[Winner_Match_1], (100 * (1 - Predictions[Match_9]))), 'Winner =', Team_Dictionary[Winner_Match_9])

    try:
        Match_10 = ('2016_%d_%d' % (Winner_Match_3, Winner_Match_4))

        if Predictions[Match_10] >= .5:
            Winner_Match_10 = Winner_Match_3
        elif Predictions[Match_10] <= .5:
            Winner_Match_10 = Winner_Match_4

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_3], (100 * Predictions[Match_10]), Team_Dictionary[Winner_Match_4], (100 * (1 - Predictions[Match_10]))), 'Winner =', Team_Dictionary[Winner_Match_10])

    except:
        Match_10 = ('2016_%d_%d' % (Winner_Match_4, Winner_Match_3))

        if Predictions[Match_10] >= .5:
            Winner_Match_10 = Winner_Match_4
        elif Predictions[Match_10] <= .5:
            Winner_Match_10 = Winner_Match_3

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_4], (100 * Predictions[Match_10]), Team_Dictionary[Winner_Match_3], (100 * (1 - Predictions[Match_10]))), 'Winner =', Team_Dictionary[Winner_Match_10])

    try:
        Match_11 = ('2016_%d_%d' % (Winner_Match_5, Winner_Match_6))

        if Predictions[Match_11] >= .5:
            Winner_Match_11 = Winner_Match_5
        elif Predictions[Match_11] <= .5:
            Winner_Match_11 = Winner_Match_6

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_5], (100 * Predictions[Match_11]), Team_Dictionary[Winner_Match_6], (100 * (1 - Predictions[Match_11]))), 'Winner =', Team_Dictionary[Winner_Match_11])

    except:
        Match_11 = ('2016_%d_%d' % (Winner_Match_6, Winner_Match_5))

        if Predictions[Match_11] >= .5:
            Winner_Match_11 = Winner_Match_6
        elif Predictions[Match_11] <= .5:
            Winner_Match_11 = Winner_Match_5

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_6], (100 * Predictions[Match_11]), Team_Dictionary[Winner_Match_5], (100 * (1 - Predictions[Match_11]))), 'Winner =', Team_Dictionary[Winner_Match_11])

    try:
        Match_12 = ('2016_%d_%d' % (Winner_Match_7, Winner_Match_8))

        if Predictions[Match_12] >= .5:
            Winner_Match_12 = Winner_Match_7
        elif Predictions[Match_12] <= .5:
            Winner_Match_12 = Winner_Match_8

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_7], (100 * Predictions[Match_12]), Team_Dictionary[Winner_Match_8], (100 * (1 - Predictions[Match_12]))), 'Winner =', Team_Dictionary[Winner_Match_12])

    except:
        Match_12 = ('2016_%d_%d' % (Winner_Match_8, Winner_Match_7))

        if Predictions[Match_12] >= .5:
            Winner_Match_12 = Winner_Match_8
        elif Predictions[Match_12] <= .5:
            Winner_Match_12 = Winner_Match_7

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_8], (100 * Predictions[Match_12]), Team_Dictionary[Winner_Match_7], (100 * (1 - Predictions[Match_12]))), 'Winner =', Team_Dictionary[Winner_Match_12])

    print ('-------------------------')

    try:
        Match_13 = ('2016_%d_%d' % (Winner_Match_10, Winner_Match_9))

        if Predictions[Match_13] >= .5:
            Winner_Match_13 = Winner_Match_10
        elif Predictions[Match_13] <= .5:
            Winner_Match_13 = Winner_Match_9

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_10], (100 * Predictions[Match_13]), Team_Dictionary[Winner_Match_9], (100 * (1 - Predictions[Match_13]))), 'Winner =', Team_Dictionary[Winner_Match_13])

    except:
        Match_13 = ('2016_%d_%d' % (Winner_Match_9, Winner_Match_10))

        if Predictions[Match_13] >= .5:
            Winner_Match_13 = Winner_Match_9
        elif Predictions[Match_13] <= .5:
            Winner_Match_13 = Winner_Match_10

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_9], (100 * Predictions[Match_13]), Team_Dictionary[Winner_Match_10], (100 * (1 - Predictions[Match_13]))), 'Winner =', Team_Dictionary[Winner_Match_13])

    try:
        Match_14 = ('2016_%d_%d' % (Winner_Match_11, Winner_Match_12))

        if Predictions[Match_14] >= .5:
            Winner_Match_14 = Winner_Match_11
        elif Predictions[Match_14] <= .5:
            Winner_Match_14 = Winner_Match_12

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_11], (100 * Predictions[Match_14]), Team_Dictionary[Winner_Match_12], (100 * (1 - Predictions[Match_14]))), 'Winner =', Team_Dictionary[Winner_Match_14])

    except:
        Match_14 = ('2016_%d_%d' % (Winner_Match_12, Winner_Match_11))

        if Predictions[Match_14] >= .5:
            Winner_Match_14 = Winner_Match_12
        elif Predictions[Match_14] <= .5:
            Winner_Match_14 = Winner_Match_11

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_12], (100 * Predictions[Match_14]), Team_Dictionary[Winner_Match_11], (100 * (1 - Predictions[Match_14]))), 'Winner =', Team_Dictionary[Winner_Match_14])

    print ('-------------------------')

    try:
        Match_15 = ('2016_%d_%d' % (Winner_Match_13, Winner_Match_14))

        if Predictions[Match_15] >= .5:
            Winner_Match_15 = Winner_Match_13
        elif Predictions[Match_15] <= .5:
            Winner_Match_15 = Winner_Match_14

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_13], (100 * Predictions[Match_15]), Team_Dictionary[Winner_Match_14], (100 * (1 - Predictions[Match_15]))), 'Winner =', Team_Dictionary[Winner_Match_15])

    except:
        Match_15 = ('2016_%d_%d' % (Winner_Match_14, Winner_Match_13))

        if Predictions[Match_15] >= .5:
            Winner_Match_15 = Winner_Match_14
        elif Predictions[Match_15] <= .5:
            Winner_Match_15 = Winner_Match_13

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_14], (100 * Predictions[Match_15]), Team_Dictionary[Winner_Match_13], (100 * (1 - Predictions[Match_15]))), 'Winner =', Team_Dictionary[Winner_Match_15])

    return (Winner_Match_15)

def South_Bracket(Bracket_South_Seeds, Predictions, Team_Dictionary):

    print ('')
    print ('')

    print ('SOUTH RESULTS')

    print ('-------------------------')
    print ('-------------------------')

    try:
        PreMatch_1 = ('2016_%d_%d' % (Bracket_South_Seeds['Y11a'], Bracket_South_Seeds['Y11b']))

        if Predictions[PreMatch_1] >= .5:
            Winner_PreMatch_1 = Bracket_South_Seeds['Y11a']
        elif Predictions[PreMatch_1] <= .5:
            Winner_PreMatch_1 = Bracket_South_Seeds['Y11b']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_South_Seeds['Y11a']], (100 * Predictions[PreMatch_1]), Team_Dictionary[Bracket_South_Seeds['Y11b']], (100 * (1 - Predictions[PreMatch_1]))), 'Winner =', Team_Dictionary[Winner_PreMatch_1])

    except:
        PreMatch_1 = ('2016_%d_%d' % (Bracket_South_Seeds['Y11b'], Bracket_South_Seeds['Y11a']))

        if Predictions[PreMatch_1] >= .5:
            Winner_PreMatch_1 = Bracket_South_Seeds['Y11b']
        elif Predictions[PreMatch_1] <= .5:
            Winner_PreMatch_1 = Bracket_South_Seeds['Y11a']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_South_Seeds['Y11b']], (100 * Predictions[PreMatch_1]), Team_Dictionary[Bracket_South_Seeds['Y11a']], (100 * (1 - Predictions[PreMatch_1]))), 'Winner =', Team_Dictionary[Winner_PreMatch_1])

    print ('-------------------------')

    try:
        Match_1 = ('2016_%d_%d' % (Bracket_South_Seeds['Y16'], Bracket_South_Seeds['Y01']))

        if Predictions[Match_1] >= .5:
            Winner_Match_1 = Bracket_South_Seeds['Y16']
        elif Predictions[Match_1] <= .5:
            Winner_Match_1 = Bracket_South_Seeds['Y01']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_South_Seeds['Y16']], (100 * (Predictions[Match_1])), Team_Dictionary[Bracket_South_Seeds['Y01']], (100 * (1 - Predictions[Match_1]))), 'Winner =', Team_Dictionary[Winner_Match_1])

    except:
        Match_1 = ('2016_%d_%d' % (Bracket_South_Seeds['Y01'], Bracket_South_Seeds['Y16']))

        if Predictions[Match_1] >= .5:
            Winner_Match_1 = Bracket_South_Seeds['Y01']
        elif Predictions[Match_1] <= .5:
            Winner_Match_1 = Bracket_South_Seeds['Y16']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_South_Seeds['Y01']], (100 * (Predictions[Match_1])), Team_Dictionary[Bracket_South_Seeds['Y16']], (100 * (1 - Predictions[Match_1]))), 'Winner =', Team_Dictionary[Winner_Match_1])

    try:
        Match_2 = ('2016_%d_%d' % (Bracket_South_Seeds['Y09'], Bracket_South_Seeds['Y08']))

        if Predictions[Match_2] >= .5:
            Winner_Match_2 = Bracket_South_Seeds['Y09']
        elif Predictions[Match_2] <= .5:
            Winner_Match_2 = Bracket_South_Seeds['Y08']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_South_Seeds['Y09']], (100 * Predictions[Match_2]), Team_Dictionary[Bracket_South_Seeds['Y08']], (100 * (1 - Predictions[Match_2]))), 'Winner =', Team_Dictionary[Winner_Match_2])

    except:
        Match_2 = ('2016_%d_%d' % (Bracket_South_Seeds['Y08'], Bracket_South_Seeds['Y09']))

        if Predictions[Match_2] >= .5:
            Winner_Match_2 = Bracket_South_Seeds['Y08']
        elif Predictions[Match_2] <= .5:
            Winner_Match_2 = Bracket_South_Seeds['Y09']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_South_Seeds['Y08']], (100 * Predictions[Match_2]), Team_Dictionary[Bracket_South_Seeds['Y09']], (100 * (1 - Predictions[Match_2]))), 'Winner =', Team_Dictionary[Winner_Match_2])

    try:
        Match_3 = ('2016_%d_%d' % (Bracket_South_Seeds['Y12'], Bracket_South_Seeds['Y05']))

        if Predictions[Match_3] >= .5:
            Winner_Match_3 = Bracket_South_Seeds['Y12']
        elif Predictions[Match_3] <= .5:
            Winner_Match_3 = Bracket_South_Seeds['Y05']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_South_Seeds['Y12']], (100 * Predictions[Match_3]), Team_Dictionary[Bracket_South_Seeds['Y05']], (100 * (1 - Predictions[Match_3]))), 'Winner =', Team_Dictionary[Winner_Match_3])

    except:
        Match_3 = ('2016_%d_%d' % (Bracket_South_Seeds['Y05'], Bracket_South_Seeds['Y12']))

        if Predictions[Match_3] >= .5:
            Winner_Match_3 = Bracket_South_Seeds['Y05']
        elif Predictions[Match_3] <= .5:
            Winner_Match_3 = Bracket_South_Seeds['Y12']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_South_Seeds['Y05']], (100 * Predictions[Match_3]), Team_Dictionary[Bracket_South_Seeds['Y12']], (100 * (1 - Predictions[Match_3]))), 'Winner =', Team_Dictionary[Winner_Match_3])

    try:
        Match_4 = ('2016_%d_%d' % (Bracket_South_Seeds['Y04'], Bracket_South_Seeds['Y13']))

        if Predictions[Match_4] >= .5:
            Winner_Match_4 = Bracket_South_Seeds['Y04']
        elif Predictions[Match_4] <= .5:
            Winner_Match_4 = Bracket_South_Seeds['Y13']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_South_Seeds['Y04']], (100 * Predictions[Match_4]), Team_Dictionary[Bracket_South_Seeds['Y13']], (100 * (1 - Predictions[Match_4]))), 'Winner =', Team_Dictionary[Winner_Match_4])

    except:
        Match_4 = ('2016_%d_%d' % (Bracket_South_Seeds['Y13'], Bracket_South_Seeds['Y04']))

        if Predictions[Match_4] >= .5:
            Winner_Match_4 = Bracket_South_Seeds['Y13']
        elif Predictions[Match_4] <= .5:
            Winner_Match_4 = Bracket_South_Seeds['Y04']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_South_Seeds['Y13']], (100 * Predictions[Match_4]), Team_Dictionary[Bracket_South_Seeds['Y04']], (100 * (1 - Predictions[Match_4]))), 'Winner =', Team_Dictionary[Winner_Match_4])

    try:
        Match_5 = ('2016_%d_%d' % (Winner_PreMatch_1, Bracket_South_Seeds['Y06']))

        if Predictions[Match_5] >= .5:
            Winner_Match_5 = Winner_PreMatch_1
        elif Predictions[Match_5] <= .5:
            Winner_Match_5 = Bracket_South_Seeds['Y06']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_PreMatch_1], (100 * Predictions[Match_5]), Team_Dictionary[Bracket_South_Seeds['Y06']], (100 * (1 - Predictions[Match_5]))), 'Winner =', Team_Dictionary[Winner_Match_5])

    except:
        Match_5 = ('2016_%d_%d' % (Bracket_South_Seeds['Y06'], Winner_PreMatch_1))

        if Predictions[Match_5] >= .5:
            Winner_Match_5 = Bracket_South_Seeds['Y06']
        elif Predictions[Match_5] <= .5:
            Winner_Match_5 = Winner_PreMatch_1

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_South_Seeds['Y06']], (100 * Predictions[Match_5]), Team_Dictionary[Winner_PreMatch_1], (100 * (1 - Predictions[Match_5]))), 'Winner =', Team_Dictionary[Winner_Match_5])

    try:
        Match_6 = ('2016_%d_%d' % (Bracket_South_Seeds['Y14'], Bracket_South_Seeds['Y03']))

        if Predictions[Match_6] >= .5:
            Winner_Match_6 = Bracket_South_Seeds['Y14']
        elif Predictions[Match_6] <= .5:
            Winner_Match_6 = Bracket_South_Seeds['Y03']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_South_Seeds['Y14']], (100 * Predictions[Match_6]), Team_Dictionary[Bracket_South_Seeds['Y03']], (100 * (1 - Predictions[Match_6]))), 'Winner =', Team_Dictionary[Winner_Match_6])

    except:
        Match_6 = ('2016_%d_%d' % (Bracket_South_Seeds['Y03'], Bracket_South_Seeds['Y14']))

        if Predictions[Match_6] >= .5:
            Winner_Match_6 = Bracket_South_Seeds['Y03']
        elif Predictions[Match_6] <= .5:
            Winner_Match_6 = Bracket_South_Seeds['Y14']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_South_Seeds['Y03']], (100 * Predictions[Match_6]), Team_Dictionary[Bracket_South_Seeds['Y14']], (100 * (1 - Predictions[Match_6]))), 'Winner =', Team_Dictionary[Winner_Match_6])

    try:
        Match_7 = ('2016_%d_%d' % (Bracket_South_Seeds['Y07'], Bracket_South_Seeds['Y10']))

        if Predictions[Match_7] >= .5:
            Winner_Match_7 = Bracket_South_Seeds['Y07']
        elif Predictions[Match_7] <= .5:
            Winner_Match_7 = Bracket_South_Seeds['Y10']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_South_Seeds['Y07']], (100 * Predictions[Match_7]), Team_Dictionary[Bracket_South_Seeds['Y10']], (100 * (1 - Predictions[Match_7]))), 'Winner =', Team_Dictionary[Winner_Match_7])

    except:
        Match_7 = ('2016_%d_%d' % (Bracket_South_Seeds['Y10'], Bracket_South_Seeds['Y07']))

        if Predictions[Match_7] >= .5:
            Winner_Match_7 = Bracket_South_Seeds['Y10']
        elif Predictions[Match_7] <= .5:
            Winner_Match_7 = Bracket_South_Seeds['Y07']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_South_Seeds['Y10']], (100 * Predictions[Match_7]), Team_Dictionary[Bracket_South_Seeds['Y07']], (100 * (1 - Predictions[Match_7]))), 'Winner =', Team_Dictionary[Winner_Match_7])

    try:
        Match_8 = ('2016_%d_%d' % (Bracket_South_Seeds['Y15'], Bracket_South_Seeds['Y02']))

        if Predictions[Match_8] >= .5:
            Winner_Match_8 = Bracket_South_Seeds['Y15']
        elif Predictions[Match_8] <= .5:
            Winner_Match_8 = Bracket_South_Seeds['Y02']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_South_Seeds['Y15']], (100 * Predictions[Match_8]), Team_Dictionary[Bracket_South_Seeds['Y02']], (100 * (1 - Predictions[Match_8]))), 'Winner =', Team_Dictionary[Winner_Match_8])

    except:
        Match_8 = ('2016_%d_%d' % (Bracket_South_Seeds['Y02'], Bracket_South_Seeds['Y15']))

        if Predictions[Match_8] >= .5:
            Winner_Match_8 = Bracket_South_Seeds['Y02']
        elif Predictions[Match_8] <= .5:
            Winner_Match_8 = Bracket_South_Seeds['Y15']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_South_Seeds['Y02']], (100 * Predictions[Match_8]), Team_Dictionary[Bracket_South_Seeds['Y15']], (100 * (1 - Predictions[Match_8]))), 'Winner =', Team_Dictionary[Winner_Match_8])

    print ('-------------------------')

    try:
        Match_9 = ('2016_%d_%d' % (Winner_Match_1, Winner_Match_2))

        if Predictions[Match_9] >= .5:
            Winner_Match_9 = Winner_Match_1
        elif Predictions[Match_9] <= .5:
            Winner_Match_9 = Winner_Match_2

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_1], (100 * Predictions[Match_9]), Team_Dictionary[Winner_Match_2], (100 * (1 - Predictions[Match_9]))), 'Winner =', Team_Dictionary[Winner_Match_9])

    except:
        Match_9 = ('2016_%d_%d' % (Winner_Match_2, Winner_Match_1))

        if Predictions[Match_9] >= .5:
            Winner_Match_9 = Winner_Match_2
        elif Predictions[Match_9] <= .5:
            Winner_Match_9 = Winner_Match_1

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_2], (100 * Predictions[Match_9]), Team_Dictionary[Winner_Match_1], (100 * (1 - Predictions[Match_9]))), 'Winner =', Team_Dictionary[Winner_Match_9])

    try:
        Match_10 = ('2016_%d_%d' % (Winner_Match_3, Winner_Match_4))

        if Predictions[Match_10] >= .5:
            Winner_Match_10 = Winner_Match_3
        elif Predictions[Match_10] <= .5:
            Winner_Match_10 = Winner_Match_4

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_3], (100 * Predictions[Match_10]), Team_Dictionary[Winner_Match_4], (100 * (1 - Predictions[Match_10]))), 'Winner =', Team_Dictionary[Winner_Match_10])

    except:
        Match_10 = ('2016_%d_%d' % (Winner_Match_4, Winner_Match_3))

        if Predictions[Match_10] >= .5:
            Winner_Match_10 = Winner_Match_4
        elif Predictions[Match_10] <= .5:
            Winner_Match_10 = Winner_Match_3

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_4], (100 * Predictions[Match_10]), Team_Dictionary[Winner_Match_3], (100 * (1 - Predictions[Match_10]))), 'Winner =', Team_Dictionary[Winner_Match_10])

    try:
        Match_11 = ('2016_%d_%d' % (Winner_Match_5, Winner_Match_6))

        if Predictions[Match_11] >= .5:
            Winner_Match_11 = Winner_Match_5
        elif Predictions[Match_11] <= .5:
            Winner_Match_11 = Winner_Match_6

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_5], (100 * Predictions[Match_11]), Team_Dictionary[Winner_Match_6], (100 * (1 - Predictions[Match_11]))), 'Winner =', Team_Dictionary[Winner_Match_11])

    except:
        Match_11 = ('2016_%d_%d' % (Winner_Match_6, Winner_Match_5))

        if Predictions[Match_11] >= .5:
            Winner_Match_11 = Winner_Match_6
        elif Predictions[Match_11] <= .5:
            Winner_Match_11 = Winner_Match_5

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_6], (100 * Predictions[Match_11]), Team_Dictionary[Winner_Match_5], (100 * (1 - Predictions[Match_11]))), 'Winner =', Team_Dictionary[Winner_Match_11])

    try:
        Match_12 = ('2016_%d_%d' % (Winner_Match_7, Winner_Match_8))

        if Predictions[Match_12] >= .5:
            Winner_Match_12 = Winner_Match_7
        elif Predictions[Match_12] <= .5:
            Winner_Match_12 = Winner_Match_8

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_7], (100 * Predictions[Match_12]), Team_Dictionary[Winner_Match_8], (100 * (1 - Predictions[Match_12]))), 'Winner =', Team_Dictionary[Winner_Match_12])

    except:
        Match_12 = ('2016_%d_%d' % (Winner_Match_8, Winner_Match_7))

        if Predictions[Match_12] >= .5:
            Winner_Match_12 = Winner_Match_8
        elif Predictions[Match_12] <= .5:
            Winner_Match_12 = Winner_Match_7

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_8], (100 * Predictions[Match_12]), Team_Dictionary[Winner_Match_7], (100 * (1 - Predictions[Match_12]))), 'Winner =', Team_Dictionary[Winner_Match_12])

    print ('-------------------------')

    try:
        Match_13 = ('2016_%d_%d' % (Winner_Match_10, Winner_Match_9))

        if Predictions[Match_13] >= .5:
            Winner_Match_13 = Winner_Match_10
        elif Predictions[Match_13] <= .5:
            Winner_Match_13 = Winner_Match_9

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_10], (100 * Predictions[Match_13]), Team_Dictionary[Winner_Match_9], (100 * (1 - Predictions[Match_13]))), 'Winner =', Team_Dictionary[Winner_Match_13])

    except:
        Match_13 = ('2016_%d_%d' % (Winner_Match_9, Winner_Match_10))

        if Predictions[Match_13] >= .5:
            Winner_Match_13 = Winner_Match_9
        elif Predictions[Match_13] <= .5:
            Winner_Match_13 = Winner_Match_10

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_9], (100 * Predictions[Match_13]), Team_Dictionary[Winner_Match_10], (100 * (1 - Predictions[Match_13]))), 'Winner =', Team_Dictionary[Winner_Match_13])

    try:
        Match_14 = ('2016_%d_%d' % (Winner_Match_11, Winner_Match_12))

        if Predictions[Match_14] >= .5:
            Winner_Match_14 = Winner_Match_11
        elif Predictions[Match_14] <= .5:
            Winner_Match_14 = Winner_Match_12

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_11], (100 * Predictions[Match_14]), Team_Dictionary[Winner_Match_12], (100 * (1 - Predictions[Match_14]))), 'Winner =', Team_Dictionary[Winner_Match_14])

    except:
        Match_14 = ('2016_%d_%d' % (Winner_Match_12, Winner_Match_11))

        if Predictions[Match_14] >= .5:
            Winner_Match_14 = Winner_Match_12
        elif Predictions[Match_14] <= .5:
            Winner_Match_14 = Winner_Match_11

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_12], (100 * Predictions[Match_14]), Team_Dictionary[Winner_Match_11], (100 * (1 - Predictions[Match_14]))), 'Winner =', Team_Dictionary[Winner_Match_14])

    print ('-------------------------')

    try:
        Match_15 = ('2016_%d_%d' % (Winner_Match_13, Winner_Match_14))

        if Predictions[Match_15] >= .5:
            Winner_Match_15 = Winner_Match_13
        elif Predictions[Match_15] <= .5:
            Winner_Match_15 = Winner_Match_14

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_13], (100 * Predictions[Match_15]), Team_Dictionary[Winner_Match_14], (100 * (1 - Predictions[Match_15]))), 'Winner =', Team_Dictionary[Winner_Match_15])

    except:
        Match_15 = ('2016_%d_%d' % (Winner_Match_14, Winner_Match_13))

        if Predictions[Match_15] >= .5:
            Winner_Match_15 = Winner_Match_14
        elif Predictions[Match_15] <= .5:
            Winner_Match_15 = Winner_Match_13

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_14], (100 * Predictions[Match_15]), Team_Dictionary[Winner_Match_13], (100 * (1 - Predictions[Match_15]))), 'Winner =', Team_Dictionary[Winner_Match_15])

    return (Winner_Match_15)

def West_Bracket(Bracket_West_Seeds, Predictions, Team_Dictionary):

    print ('')
    print ('')

    print ('WEST RESULTS')

    print ('-------------------------')
    print ('-------------------------')

    try:
        PreMatch_1 = ('2016_%d_%d' % (Bracket_West_Seeds['Z16a'], Bracket_West_Seeds['Z16b']))

        if Predictions[PreMatch_1] >= .5:
            Winner_PreMatch_1 = Bracket_West_Seeds['Z16a']
        elif Predictions[PreMatch_1] <= .5:
            Winner_PreMatch_1 = Bracket_West_Seeds['Z16b']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_West_Seeds['Z16a']], (100 * Predictions[PreMatch_1]), Team_Dictionary[Bracket_West_Seeds['Z16b']], (100 * (1 - Predictions[PreMatch_1]))), 'Winner =', Team_Dictionary[Winner_PreMatch_1])

    except:
        PreMatch_1 = ('2016_%d_%d' % (Bracket_West_Seeds['Z16b'], Bracket_West_Seeds['Z16a']))

        if Predictions[PreMatch_1] >= .5:
            Winner_PreMatch_1 = Bracket_West_Seeds['Z16b']
        elif Predictions[PreMatch_1] <= .5:
            Winner_PreMatch_1 = Bracket_West_Seeds['Z16a']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_West_Seeds['Z16b']], (100 * Predictions[PreMatch_1]), Team_Dictionary[Bracket_West_Seeds['Z16a']], (100 * (1 - Predictions[PreMatch_1]))), 'Winner =', Team_Dictionary[Winner_PreMatch_1])

    print ('-------------------------')

    try:
        Match_1 = ('2016_%d_%d' % (Winner_PreMatch_1, Bracket_West_Seeds['Z01']))

        if Predictions[Match_1] >= .5:
            Winner_Match_1 = Winner_PreMatch_1
        elif Predictions[Match_1] <= .5:
            Winner_Match_1 = Bracket_West_Seeds['Z01']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_PreMatch_1], (100 * (Predictions[Match_1])), Team_Dictionary[Bracket_West_Seeds['Z01']], (100 * (1 - Predictions[Match_1]))), 'Winner =', Team_Dictionary[Winner_Match_1])

    except:
        Match_1 = ('2016_%d_%d' % (Bracket_West_Seeds['Z01'], Winner_PreMatch_1))

        if Predictions[Match_1] >= .5:
            Winner_Match_1 = Bracket_West_Seeds['Z01']
        elif Predictions[Match_1] <= .5:
            Winner_Match_1 = Winner_PreMatch_1

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_West_Seeds['Z01']], (100 * (Predictions[Match_1])), Team_Dictionary[Winner_PreMatch_1], (100 * (1 - Predictions[Match_1]))), 'Winner =', Team_Dictionary[Winner_Match_1])

    try:
        Match_2 = ('2016_%d_%d' % (Bracket_West_Seeds['Z09'], Bracket_West_Seeds['Z08']))

        if Predictions[Match_2] >= .5:
            Winner_Match_2 = Bracket_West_Seeds['Z09']
        elif Predictions[Match_2] <= .5:
            Winner_Match_2 = Bracket_West_Seeds['Z08']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_West_Seeds['Z09']], (100 * Predictions[Match_2]), Team_Dictionary[Bracket_West_Seeds['Z08']], (100 * (1 - Predictions[Match_2]))), 'Winner =', Team_Dictionary[Winner_Match_2])

    except:
        Match_2 = ('2016_%d_%d' % (Bracket_West_Seeds['Z08'], Bracket_West_Seeds['Z09']))

        if Predictions[Match_2] >= .5:
            Winner_Match_2 = Bracket_West_Seeds['Z08']
        elif Predictions[Match_2] <= .5:
            Winner_Match_2 = Bracket_West_Seeds['Z09']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_West_Seeds['Z08']], (100 * Predictions[Match_2]), Team_Dictionary[Bracket_West_Seeds['Z09']], (100 * (1 - Predictions[Match_2]))), 'Winner =', Team_Dictionary[Winner_Match_2])

    try:
        Match_3 = ('2016_%d_%d' % (Bracket_West_Seeds['Z12'], Bracket_West_Seeds['Z05']))

        if Predictions[Match_3] >= .5:
            Winner_Match_3 = Bracket_West_Seeds['Z12']
        elif Predictions[Match_3] <= .5:
            Winner_Match_3 = Bracket_West_Seeds['Z05']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_West_Seeds['Z12']], (100 * Predictions[Match_3]), Team_Dictionary[Bracket_West_Seeds['Z05']], (100 * (1 - Predictions[Match_3]))), 'Winner =', Team_Dictionary[Winner_Match_3])

    except:
        Match_3 = ('2016_%d_%d' % (Bracket_West_Seeds['Z05'], Bracket_West_Seeds['Z12']))

        if Predictions[Match_3] >= .5:
            Winner_Match_3 = Bracket_West_Seeds['Z05']
        elif Predictions[Match_3] <= .5:
            Winner_Match_3 = Bracket_West_Seeds['Z12']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_West_Seeds['Z05']], (100 * Predictions[Match_3]), Team_Dictionary[Bracket_West_Seeds['Z12']], (100 * (1 - Predictions[Match_3]))), 'Winner =', Team_Dictionary[Winner_Match_3])

    try:
        Match_4 = ('2016_%d_%d' % (Bracket_West_Seeds['Z04'], Bracket_West_Seeds['Z13']))

        if Predictions[Match_4] >= .5:
            Winner_Match_4 = Bracket_West_Seeds['Z04']
        elif Predictions[Match_4] <= .5:
            Winner_Match_4 = Bracket_West_Seeds['Z13']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_West_Seeds['Z04']], (100 * Predictions[Match_4]), Team_Dictionary[Bracket_West_Seeds['Z13']], (100 * (1 - Predictions[Match_4]))), 'Winner =', Team_Dictionary[Winner_Match_4])

    except:
        Match_4 = ('2016_%d_%d' % (Bracket_West_Seeds['Z13'], Bracket_West_Seeds['Z04']))

        if Predictions[Match_4] >= .5:
            Winner_Match_4 = Bracket_West_Seeds['Z13']
        elif Predictions[Match_4] <= .5:
            Winner_Match_4 = Bracket_West_Seeds['Z04']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_West_Seeds['Z13']], (100 * Predictions[Match_4]), Team_Dictionary[Bracket_West_Seeds['Z04']], (100 * (1 - Predictions[Match_4]))), 'Winner =', Team_Dictionary[Winner_Match_4])

    try:
        Match_5 = ('2016_%d_%d' % (Bracket_West_Seeds['Z11'], Bracket_West_Seeds['Z06']))

        if Predictions[Match_5] >= .5:
            Winner_Match_5 = Bracket_West_Seeds['Z11']
        elif Predictions[Match_5] <= .5:
            Winner_Match_5 = Bracket_West_Seeds['Z06']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_West_Seeds['Z11']], (100 * Predictions[Match_5]), Team_Dictionary[Bracket_West_Seeds['Z06']], (100 * (1 - Predictions[Match_5]))), 'Winner =', Team_Dictionary[Winner_Match_5])

    except:
        Match_5 = ('2016_%d_%d' % (Bracket_West_Seeds['Z06'], Bracket_West_Seeds['Z11']))

        if Predictions[Match_5] >= .5:
            Winner_Match_5 = Bracket_West_Seeds['Z06']
        elif Predictions[Match_5] <= .5:
            Winner_Match_5 = Bracket_West_Seeds['Z11']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_West_Seeds['Z06']], (100 * Predictions[Match_5]), Team_Dictionary[Bracket_West_Seeds['Z11']], (100 * (1 - Predictions[Match_5]))), 'Winner =', Team_Dictionary[Winner_Match_5])

    try:
        Match_6 = ('2016_%d_%d' % (Bracket_West_Seeds['Z14'], Bracket_West_Seeds['Z03']))

        if Predictions[Match_6] >= .5:
            Winner_Match_6 = Bracket_West_Seeds['Z14']
        elif Predictions[Match_6] <= .5:
            Winner_Match_6 = Bracket_West_Seeds['Z03']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_West_Seeds['Z14']], (100 * Predictions[Match_6]), Team_Dictionary[Bracket_West_Seeds['Z03']], (100 * (1 - Predictions[Match_6]))), 'Winner =', Team_Dictionary[Winner_Match_6])

    except:
        Match_6 = ('2016_%d_%d' % (Bracket_West_Seeds['Z03'], Bracket_West_Seeds['Z14']))

        if Predictions[Match_6] >= .5:
            Winner_Match_6 = Bracket_West_Seeds['Z03']
        elif Predictions[Match_6] <= .5:
            Winner_Match_6 = Bracket_West_Seeds['Z14']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_West_Seeds['Z03']], (100 * Predictions[Match_6]), Team_Dictionary[Bracket_West_Seeds['Z14']], (100 * (1 - Predictions[Match_6]))), 'Winner =', Team_Dictionary[Winner_Match_6])

    try:
        Match_7 = ('2016_%d_%d' % (Bracket_West_Seeds['Z07'], Bracket_West_Seeds['Z10']))

        if Predictions[Match_7] >= .5:
            Winner_Match_7 = Bracket_West_Seeds['Z07']
        elif Predictions[Match_7] <= .5:
            Winner_Match_7 = Bracket_West_Seeds['Z10']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_West_Seeds['Z07']], (100 * Predictions[Match_7]), Team_Dictionary[Bracket_West_Seeds['Z10']], (100 * (1 - Predictions[Match_7]))), 'Winner =', Team_Dictionary[Winner_Match_7])

    except:
        Match_7 = ('2016_%d_%d' % (Bracket_West_Seeds['Z10'], Bracket_West_Seeds['Z07']))

        if Predictions[Match_7] >= .5:
            Winner_Match_7 = Bracket_West_Seeds['Z10']
        elif Predictions[Match_7] <= .5:
            Winner_Match_7 = Bracket_West_Seeds['Z07']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_West_Seeds['Z10']], (100 * Predictions[Match_7]), Team_Dictionary[Bracket_West_Seeds['Z07']], (100 * (1 - Predictions[Match_7]))), 'Winner =', Team_Dictionary[Winner_Match_7])

    try:
        Match_8 = ('2016_%d_%d' % (Bracket_West_Seeds['Z15'], Bracket_West_Seeds['Z02']))

        if Predictions[Match_8] >= .5:
            Winner_Match_8 = Bracket_West_Seeds['Z15']
        elif Predictions[Match_8] <= .5:
            Winner_Match_8 = Bracket_West_Seeds['Z02']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_West_Seeds['Z15']], (100 * Predictions[Match_8]), Team_Dictionary[Bracket_West_Seeds['Z02']], (100 * (1 - Predictions[Match_8]))), 'Winner =', Team_Dictionary[Winner_Match_8])

    except:
        Match_8 = ('2016_%d_%d' % (Bracket_West_Seeds['Z02'], Bracket_West_Seeds['Z15']))

        if Predictions[Match_8] >= .5:
            Winner_Match_8 = Bracket_West_Seeds['Z02']
        elif Predictions[Match_8] <= .5:
            Winner_Match_8 = Bracket_West_Seeds['Z15']

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Bracket_West_Seeds['Z02']], (100 * Predictions[Match_8]), Team_Dictionary[Bracket_West_Seeds['Z15']], (100 * (1 - Predictions[Match_8]))), 'Winner =', Team_Dictionary[Winner_Match_8])

    print ('-------------------------')

    try:
        Match_9 = ('2016_%d_%d' % (Winner_Match_1, Winner_Match_2))

        if Predictions[Match_9] >= .5:
            Winner_Match_9 = Winner_Match_1
        elif Predictions[Match_9] <= .5:
            Winner_Match_9 = Winner_Match_2

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_1], (100 * Predictions[Match_9]), Team_Dictionary[Winner_Match_2], (100 * (1 - Predictions[Match_9]))), 'Winner =', Team_Dictionary[Winner_Match_9])

    except:
        Match_9 = ('2016_%d_%d' % (Winner_Match_2, Winner_Match_1))

        if Predictions[Match_9] >= .5:
            Winner_Match_9 = Winner_Match_2
        elif Predictions[Match_9] <= .5:
            Winner_Match_9 = Winner_Match_1

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_2], (100 * Predictions[Match_9]), Team_Dictionary[Winner_Match_1], (100 * (1 - Predictions[Match_9]))), 'Winner =', Team_Dictionary[Winner_Match_9])

    try:
        Match_10 = ('2016_%d_%d' % (Winner_Match_3, Winner_Match_4))

        if Predictions[Match_10] >= .5:
            Winner_Match_10 = Winner_Match_3
        elif Predictions[Match_10] <= .5:
            Winner_Match_10 = Winner_Match_4

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_3], (100 * Predictions[Match_10]), Team_Dictionary[Winner_Match_4], (100 * (1 - Predictions[Match_10]))), 'Winner =', Team_Dictionary[Winner_Match_10])

    except:
        Match_10 = ('2016_%d_%d' % (Winner_Match_4, Winner_Match_3))

        if Predictions[Match_10] >= .5:
            Winner_Match_10 = Winner_Match_4
        elif Predictions[Match_10] <= .5:
            Winner_Match_10 = Winner_Match_3

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_4], (100 * Predictions[Match_10]), Team_Dictionary[Winner_Match_3], (100 * (1 - Predictions[Match_10]))), 'Winner =', Team_Dictionary[Winner_Match_10])

    try:
        Match_11 = ('2016_%d_%d' % (Winner_Match_5, Winner_Match_6))

        if Predictions[Match_11] >= .5:
            Winner_Match_11 = Winner_Match_5
        elif Predictions[Match_11] <= .5:
            Winner_Match_11 = Winner_Match_6

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_5], (100 * Predictions[Match_11]), Team_Dictionary[Winner_Match_6], (100 * (1 - Predictions[Match_11]))), 'Winner =', Team_Dictionary[Winner_Match_11])

    except:
        Match_11 = ('2016_%d_%d' % (Winner_Match_6, Winner_Match_5))

        if Predictions[Match_11] >= .5:
            Winner_Match_11 = Winner_Match_6
        elif Predictions[Match_11] <= .5:
            Winner_Match_11 = Winner_Match_5

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_6], (100 * Predictions[Match_11]), Team_Dictionary[Winner_Match_5], (100 * (1 - Predictions[Match_11]))), 'Winner =', Team_Dictionary[Winner_Match_11])

    try:
        Match_12 = ('2016_%d_%d' % (Winner_Match_7, Winner_Match_8))

        if Predictions[Match_12] >= .5:
            Winner_Match_12 = Winner_Match_7
        elif Predictions[Match_12] <= .5:
            Winner_Match_12 = Winner_Match_8

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_7], (100 * Predictions[Match_12]), Team_Dictionary[Winner_Match_8], (100 * (1 - Predictions[Match_12]))), 'Winner =', Team_Dictionary[Winner_Match_12])

    except:
        Match_12 = ('2016_%d_%d' % (Winner_Match_8, Winner_Match_7))

        if Predictions[Match_12] >= .5:
            Winner_Match_12 = Winner_Match_8
        elif Predictions[Match_12] <= .5:
            Winner_Match_12 = Winner_Match_7

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_8], (100 * Predictions[Match_12]), Team_Dictionary[Winner_Match_7], (100 * (1 - Predictions[Match_12]))), 'Winner =', Team_Dictionary[Winner_Match_12])

    print ('-------------------------')

    try:
        Match_13 = ('2016_%d_%d' % (Winner_Match_10, Winner_Match_9))

        if Predictions[Match_13] >= .5:
            Winner_Match_13 = Winner_Match_10
        elif Predictions[Match_13] <= .5:
            Winner_Match_13 = Winner_Match_9

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_10], (100 * Predictions[Match_13]), Team_Dictionary[Winner_Match_9], (100 * (1 - Predictions[Match_13]))), 'Winner =', Team_Dictionary[Winner_Match_13])

    except:
        Match_13 = ('2016_%d_%d' % (Winner_Match_9, Winner_Match_10))

        if Predictions[Match_13] >= .5:
            Winner_Match_13 = Winner_Match_9
        elif Predictions[Match_13] <= .5:
            Winner_Match_13 = Winner_Match_10

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_9], (100 * Predictions[Match_13]), Team_Dictionary[Winner_Match_10], (100 * (1 - Predictions[Match_13]))), 'Winner =', Team_Dictionary[Winner_Match_13])

    try:
        Match_14 = ('2016_%d_%d' % (Winner_Match_11, Winner_Match_12))

        if Predictions[Match_14] >= .5:
            Winner_Match_14 = Winner_Match_11
        elif Predictions[Match_14] <= .5:
            Winner_Match_14 = Winner_Match_12

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_11], (100 * Predictions[Match_14]), Team_Dictionary[Winner_Match_12], (100 * (1 - Predictions[Match_14]))), 'Winner =', Team_Dictionary[Winner_Match_14])

    except:
        Match_14 = ('2016_%d_%d' % (Winner_Match_12, Winner_Match_11))

        if Predictions[Match_14] >= .5:
            Winner_Match_14 = Winner_Match_12
        elif Predictions[Match_14] <= .5:
            Winner_Match_14 = Winner_Match_11

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_12], (100 * Predictions[Match_14]), Team_Dictionary[Winner_Match_11], (100 * (1 - Predictions[Match_14]))), 'Winner =', Team_Dictionary[Winner_Match_14])

    print ('-------------------------')

    try:
        Match_15 = ('2016_%d_%d' % (Winner_Match_13, Winner_Match_14))

        if Predictions[Match_15] >= .5:
            Winner_Match_15 = Winner_Match_13
        elif Predictions[Match_15] <= .5:
            Winner_Match_15 = Winner_Match_14

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_13], (100 * Predictions[Match_15]), Team_Dictionary[Winner_Match_14], (100 * (1 - Predictions[Match_15]))), 'Winner =', Team_Dictionary[Winner_Match_15])

    except:
        Match_15 = ('2016_%d_%d' % (Winner_Match_14, Winner_Match_13))

        if Predictions[Match_15] >= .5:
            Winner_Match_15 = Winner_Match_14
        elif Predictions[Match_15] <= .5:
            Winner_Match_15 = Winner_Match_13

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_14], (100 * Predictions[Match_15]), Team_Dictionary[Winner_Match_13], (100 * (1 - Predictions[Match_15]))), 'Winner =', Team_Dictionary[Winner_Match_15])

    return (Winner_Match_15)

def Final_Games(Winner_East, Winner_Midwest, Winner_South, Winner_West, Predictions, Team_Dictionary):

    print ('')
    print ('')

    print ('FINAL FOUR')

    print ('-------------------------')
    print ('-------------------------')

    try:
        Match_1 = ('2016_%d_%d' % (Winner_Midwest, Winner_East))

        if Predictions[Match_1] >= .5:
            Winner_Match_1 = Winner_Midwest
        elif Predictions[Match_1] <= .5:
            Winner_Match_1 = Winner_East

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Midwest], (100 * Predictions[Match_1]), Team_Dictionary[Winner_East], (100 * (1 - Predictions[Match_1]))), 'Winner =', Team_Dictionary[Winner_Match_1])

    except:
        Match_1 = ('2016_%d_%d' % (Winner_East, Winner_Midwest))

        if Predictions[Match_1] >= .5:
            Winner_Match_1 = Winner_East
        elif Predictions[Match_1] <= .5:
            Winner_Match_1 = Winner_Midwest

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_East], (100 * Predictions[Match_1]), Team_Dictionary[Winner_Midwest], (100 * (1 - Predictions[Match_1]))), 'Winner =', Team_Dictionary[Winner_Match_1])

    try:
        Match_2 = ('2016_%d_%d' % (Winner_South, Winner_West))

        if Predictions[Match_2] >= .5:
            Winner_Match_2 = Winner_South
        elif Predictions[Match_2] <= .5:
            Winner_Match_2 = Winner_West

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_South], (100 * Predictions[Match_2]), Team_Dictionary[Winner_West], (100 * (1 - Predictions[Match_2]))), 'Winner =', Team_Dictionary[Winner_Match_2])

    except:
        Match_2 = ('2016_%d_%d' % (Winner_West, Winner_South))

        if Predictions[Match_2] >= .5:
            Winner_Match_2 = Winner_West
        elif Predictions[Match_2] <= .5:
            Winner_Match_2 = Winner_South

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_West], (100 * Predictions[Match_2]), Team_Dictionary[Winner_South], (100 * (1 - Predictions[Match_2]))), 'Winner =', Team_Dictionary[Winner_Match_2])

    print ('')
    print ('')

    print ('CHAMPIONSHIP')

    print ('-------------------------')
    print ('-------------------------')

    try:
        Championship = ('2016_%d_%d' % (Winner_Match_2, Winner_Match_1))

        if Predictions[Championship] >= .5:
            Winner_Championship = Winner_Match_2
        elif Predictions[Championship] <= .5:
            Winner_Championship = Winner_Match_1

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_2], (100 * Predictions[Championship]), Team_Dictionary[Winner_Match_1], (100 * (1 - Predictions[Championship]))), 'Winner =', Team_Dictionary[Winner_Championship])

    except:
        Championship = ('2016_%d_%d' % (Winner_Match_1, Winner_Match_2))

        if Predictions[Championship] >= .5:
            Winner_Championship = Winner_Match_1
        elif Predictions[Championship] <= .5:
            Winner_Championship = Winner_Match_2

        print ('%s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Winner_Match_1], (100 * Predictions[Championship]), Team_Dictionary[Winner_Match_2], (100 * (1 - Predictions[Championship]))), 'Winner =', Team_Dictionary[Winner_Championship])

    return (Winner_Championship)

Teams_Array = np.asarray(Bracket_Teams)
Teams_List_1 = list(Teams_Array[:,0])
Teams_List_2 = list(Teams_Array[:,1])

Team_Dictionary = dict(zip(Teams_List_1, Teams_List_2))

Bracket_2016 = np.asarray(Split_Array(Bracket_Seeds, 2016))
Bracket_East_2016 = np.asarray(Split_Bracket(Bracket_2016, 'W'))
Bracket_Midwest_2016 = np.asarray(Split_Bracket(Bracket_2016, 'X'))
Bracket_South_2016 = np.asarray(Split_Bracket(Bracket_2016, 'Y'))
Bracket_West_2016 = np.asarray(Split_Bracket(Bracket_2016, 'Z'))

Bracket_East_Keys = Bracket_East_2016[:,1]
Bracket_Midwest_Keys = Bracket_Midwest_2016[:,1]
Bracket_South_Keys = Bracket_South_2016[:,1]
Bracket_West_Keys = Bracket_West_2016[:,1]

Bracket_East_Values = Bracket_East_2016[:,2]
Bracket_Midwest_Values = Bracket_Midwest_2016[:,2]
Bracket_South_Values = Bracket_South_2016[:,2]
Bracket_West_Values = Bracket_West_2016[:,2]

Bracket_East_Seeds = dict(zip(Bracket_East_Keys, Bracket_East_Values))
Bracket_Midwest_Seeds = dict(zip(Bracket_Midwest_Keys, Bracket_Midwest_Values))
Bracket_South_Seeds = dict(zip(Bracket_South_Keys, Bracket_South_Values))
Bracket_West_Seeds = dict(zip(Bracket_West_Keys, Bracket_West_Values))

Predictions = np.asarray(Predictions)
Predictions_Keys = Predictions[:,0]
Predictions_Values = Predictions[:,1]
Predictions = dict(zip(Predictions_Keys, Predictions_Values))

print ('')

Winner_East = East_Bracket(Bracket_East_Seeds, Predictions, Team_Dictionary)
Winner_Midwest = Midwest_Bracket(Bracket_Midwest_Seeds, Predictions, Team_Dictionary)
Winner_South = South_Bracket(Bracket_South_Seeds, Predictions, Team_Dictionary)
Winner_West = West_Bracket(Bracket_West_Seeds, Predictions, Team_Dictionary)
Winner_Championship = Final_Games(Winner_East, Winner_Midwest, Winner_South, Winner_West, Predictions, Team_Dictionary)