__author__ = 'Nick Sarris'

import numpy as np
import pandas as pd
import collections
import scipy as sp
import pylab as pl

import csv
import operator
import timeit

from scipy.optimize import curve_fit

Teams = pd.read_csv('~/Desktop/Massey Predictor/Data_Results/Teams/Teams.csv', encoding = "ISO-8859-1")

NetProphet_Ratings = pd.read_csv('~/Desktop/Massey Predictor/Data_Results/Net Prophet/NetProphet Probabilities.csv')
NetProphet_Ratings_2016 = pd.read_csv('~/Desktop/Massey Predictor/Data_Results/Net Prophet/NetProphet Probabilities (2016).csv')

RegularSCR = pd.read_csv('~/Desktop/Massey Predictor/Data_Results/Results/RegularSeasonCompactResults.csv', encoding = "ISO-8859-1")
TourneyCR = pd.read_csv('~/Desktop/Massey Predictor/Data_Results/Results/TourneyCompactResults.csv', encoding = "ISO-8859-1")
MasseyOrdinals = pd.read_csv('~/Desktop/Massey Predictor/Data_Results/Results/MasseyOrdinals.csv', encoding = "ISO-8859-1")
MasseyOrdinals_2016 = pd.read_csv('~/Desktop/Massey Predictor/Data_Results/Results/MasseyOrdinals (2016).csv', encoding = "ISO-8859-1")

Submission = pd.read_csv('~/Desktop/Massey Predictor/Data_Results/Submission/SubmissionFormat.csv', encoding = "ISO-8859-1")
Submission_2016 = pd.read_csv('~/Desktop/Massey Predictor/Data_Results/Submission/SubmissionFormat (2016).csv', encoding = "ISO-8859-1")
Final_Submission = pd.read_csv('~/Desktop/Massey Predictor/Data_Results/Submission/Submission.csv', encoding = "ISO-8859-1")
Final_Submission_2016 = pd.read_csv('~/Desktop/Massey Predictor/Data_Results/Submission/Submission (2016).csv', encoding = "ISO-8859-1")

print ('')
print ('Running Massey_Model_Final.py...')

def Split_Array(Input, Season):

    Input_Array = np.asarray(Input)

    list_1 = []
    for a in Input_Array:
        if a[0] == Season:
            list.append(list_1, a)

    return list_1

def Initial_Elo_Rating(Team_List):

    Initial_Elo_Rating = [1500.0]

    IER_Teams = []
    for x, a in enumerate(Team_List):
        IER_Teams.append((a, Initial_Elo_Rating[x % len(Initial_Elo_Rating)]))

    IER_Dictionary = dict(IER_Teams)
    return IER_Dictionary

def Elo_Ratings(IER_Dictionary, RegularSCR_Year):

    IER_Dictionary_New = IER_Dictionary

    for x in RegularSCR_Year:

        Winner = x[2]
        Loser = x[4]
        Advantage = x[6]

        Win_Score = x[3]
        Lose_Score = x[5]

        X_Score = 12

        if Advantage == ('H'):
            Margin_of_Victory = (((Win_Score - Lose_Score + 1) ** .8)/(7.5 + (.006 * (IER_Dictionary[Winner] - IER_Dictionary[Loser]))))
            Expected_Result_Winner = X_Score * Margin_of_Victory * (1-(1/(1+(10**(-((IER_Dictionary[Winner] + 100) - IER_Dictionary[Loser])/400)))))
            Expected_Result_Loser = X_Score * Margin_of_Victory * (0-(1/(1+(10**(-(IER_Dictionary[Loser] - (IER_Dictionary[Winner] + 100))/400)))))

        elif Advantage == ('A'):
            Margin_of_Victory = (((Win_Score - Lose_Score + 1) ** .8)/(7.5 + (.006 * (IER_Dictionary[Winner] - IER_Dictionary[Loser]))))
            Expected_Result_Winner = X_Score * Margin_of_Victory * (1-(1/(1+(10**(-(IER_Dictionary[Winner] - (IER_Dictionary[Loser] + 100))/400)))))
            Expected_Result_Loser = X_Score * Margin_of_Victory * (0-(1/(1+(10**(-((IER_Dictionary[Loser] + 100) - IER_Dictionary[Winner])/400)))))

        elif Advantage == ('N'):
            Margin_of_Victory = (((Win_Score - Lose_Score + 1) ** .8)/(7.5 + (.006 * (IER_Dictionary[Winner] - IER_Dictionary[Loser]))))
            Expected_Result_Winner = X_Score * Margin_of_Victory * (1-(1/(1+(10**(-(IER_Dictionary[Winner] - IER_Dictionary[Loser])/400)))))
            Expected_Result_Loser = X_Score * Margin_of_Victory * (0-(1/(1+(10**(-(IER_Dictionary[Loser] - IER_Dictionary[Winner])/400)))))

        Final_ER_Winner = IER_Dictionary[Winner] + (Expected_Result_Winner)
        Final_ER_Loser = IER_Dictionary[Loser] + (Expected_Result_Loser)
        IER_Dictionary_New[Winner] = (Final_ER_Winner)
        IER_Dictionary_New[Loser] = (Final_ER_Loser)

    return IER_Dictionary_New

def K_Score(TourneyCR_Year):

    K_Multiplier = dict(collections.Counter(TourneyCR_Year[:,2]))

    for x in TourneyCR_Year:

        Winner = x[2]
        K_Multiplier[Winner] = (1 + K_Multiplier[Winner])

    return K_Multiplier

def Elo_Ratings_Tourney(IER_Dictionary, RegularSCR_Year, K_Score_Year):

    IER_Dictionary_New = IER_Dictionary

    for x in RegularSCR_Year:

        Winner = x[2]
        Loser = x[4]

        Win_Score = x[3]
        Lose_Score = x[5]

        X_Score = 20

        Margin_of_Victory = (((Win_Score - Lose_Score + 1) ** .8)/(7.5 + (.006 * (IER_Dictionary[Winner] - IER_Dictionary[Loser]))))

        Expected_Result_Winner = K_Score_Year[Winner] * X_Score * Margin_of_Victory * (1-(1/(1+(10**(-(IER_Dictionary[Winner] - IER_Dictionary[Loser])/400)))))
        Expected_Result_Loser = K_Score_Year[Winner] * X_Score * Margin_of_Victory * (0-(1/(1+(10**(-(IER_Dictionary[Loser] - IER_Dictionary[Winner])/400)))))

        Final_ER_Winner = IER_Dictionary[Winner] + (Expected_Result_Winner)
        Final_ER_Loser = IER_Dictionary[Loser] + (Expected_Result_Loser)
        IER_Dictionary_New[Winner] = (Final_ER_Winner)
        IER_Dictionary_New[Loser] = (Final_ER_Loser)

    return IER_Dictionary_New

def Elo_Rating_Depreciation(IER_Dictionary):

    Initial_Elo_Rating = 1500.0
    for x in IER_Dictionary:
        IER_Dictionary[x] = (1 * (IER_Dictionary[x])) + (0 * Initial_Elo_Rating)
    return IER_Dictionary

def Power_Rankings(MasseyOrdinals, Team_List):

    Initial_Ranking = [0.0]

    Massey_Teams = []
    for x, a in enumerate(Team_List):
        Massey_Teams.append((a, Initial_Ranking[x % len(Initial_Ranking)]))

    MasseyOrdinals_Dict = dict(Massey_Teams)

    Massey_Counter = dict(collections.Counter(MasseyOrdinals[:,0]))

    for x in Team_List:
        if x in Massey_Counter:
            pass
        else:
            Massey_Counter[x] = 0.0

    for x in MasseyOrdinals:
        MasseyOrdinals_Dict[x[0]] = (MasseyOrdinals_Dict[x[0]] + (100 - (4 * (np.log(x[1] + 1))) - (x[1])/22))

    for x in MasseyOrdinals_Dict:
        try:
            MasseyOrdinals_Dict[x] = MasseyOrdinals_Dict[x] / Massey_Counter[x]
        except ZeroDivisionError:
            MasseyOrdinals_Dict[x] = 0.0
    return MasseyOrdinals_Dict

def GameData(RegularSCR):

    Games = []
    Minimum_1 = (min(RegularSCR[:,2]))
    Minimum_2 = (min(RegularSCR[:,4]))

    if Minimum_1 != Minimum_2:
        Minimum_1 = min(Minimum_1, Minimum_2)
        return Minimum_1

    for x in RegularSCR:
        list.append(Games, (x[2] - Minimum_1, x[4] - Minimum_1, (x[3]- x[5])))
    return Games

def BuildGames(Games_Year):

    Team_List_1 = []
    Team_List_2 = []

    for x in Games_Year:
        list.append(Team_List_1, x[0])
        list.append(Team_List_2, x[1])

    Maximum_1 = (max(Team_List_1))
    Maximum_2 = (max(Team_List_2))

    if Maximum_1 != Maximum_2:
        Maximum_1 = max(Maximum_1, Maximum_2)

    Matrix = np.zeros([len(Games_Year), Maximum_1 + 1])

    row = 0

    for x in Games_Year:
        Matrix[row, x[0]] = 1
        Matrix[row, x[1]] = -1
        row += 1

    return Matrix

def BuildOutcomes(Games_Year):

    Matrix = np.zeros([len(Games_Year)])

    row = 0

    for x in Games_Year:
        Matrix[row] = x[2]
        row += 1

    return Matrix

def Prediction_Array(Games_Year, RegularSCR_Year, Games_LS_Year, Team_1, Team_2):

    Team_List_1 = []
    Team_List_2 = []

    for x in Games_Year:
        list.append(Team_List_1, x[0])
        list.append(Team_List_2, x[1])

    Maximum_1 = (max(Team_List_1))
    Maximum_2 = (max(Team_List_2))

    if Maximum_1 != Maximum_2:
        Maximum_1 = max(Maximum_1, Maximum_2)

    Minimum_1 = (min(RegularSCR_Year[:,2]))
    Minimum_2 = (min(RegularSCR_Year[:,4]))

    if Minimum_1 != Minimum_2:
        Minimum_1 = min(Minimum_1, Minimum_2)

    Matrix = np.zeros(Maximum_1 + 1)

    T1 = [(Team_1 - Minimum_1)]
    T2 = [(Team_2 - Minimum_1)]
    Matrix[T1] = 1
    Matrix[T2] = -1

    return (Matrix.dot(Games_LS_Year))

def Probabilities(Submission_Year, Games_Year, RegularSCR_Year, Games_LS_Year):

    Probabilities_Year = []
    for x in Submission_Year:
        Team_1 = x[2]
        Team_2 = x[4]

        Spread = Prediction_Array(Games_Year, RegularSCR_Year, Games_LS_Year, Team_1, Team_2)
        list.append(Probabilities_Year, int(round(Spread)))
    return Probabilities_Year

def TProbabilities(Submission_Year, Games_Year, RegularSCR_Year, Games_LS_Year):

    Probabilities_Year = []
    for x in Submission_Year:
        Team_1 = x[1]
        Team_2 = x[2]

        Spread = Prediction_Array(Games_Year, RegularSCR_Year, Games_LS_Year, Team_1, Team_2)
        list.append(Probabilities_Year, int(round(Spread)))
    return Probabilities_Year

def GameValues(Wins_1990, Wins_1991, Wins_1992, Wins_1993, Wins_1994, Wins_1995, Wins_1996, Wins_1997, Wins_1998, Wins_1999, Wins_2000,
               Wins_2001, Wins_2002, Wins_2003, Wins_2004, Wins_2005, Wins_2006, Wins_2007, Wins_2008, Wins_2009, Wins_2010, Wins_2011,
               Wins_2012, Wins_2013, Wins_2014, Wins_2015, Losses_1990, Losses_1991, Losses_1992, Losses_1993, Losses_1994, Losses_1995,
               Losses_1996, Losses_1997, Losses_1998, Losses_1999, Losses_2000, Losses_2001, Losses_2002, Losses_2003, Losses_2004, Losses_2005,
               Losses_2006, Losses_2007, Losses_2008, Losses_2009, Losses_2010, Losses_2011, Losses_2012, Losses_2013, Losses_2014, Losses_2015):

    Game_Values = dict()

    for x in Wins_1990:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(1)

    for x in Wins_1991:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(1)

    for x in Wins_1992:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(1)

    for x in Wins_1993:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(1)

    for x in Wins_1994:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(1)

    for x in Wins_1995:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(1)

    for x in Wins_1996:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(1)

    for x in Wins_1997:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(1)

    for x in Wins_1998:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(1)

    for x in Wins_1999:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(1)

    for x in Wins_2000:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(1)

    for x in Wins_2001:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(1)

    for x in Wins_2002:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(1)

    for x in Wins_2003:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(1)

    for x in Wins_2004:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(1)

    for x in Wins_2005:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(1)

    for x in Wins_2006:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(1)

    for x in Wins_2007:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(1)

    for x in Wins_2008:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(1)

    for x in Wins_2009:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(1)

    for x in Wins_2010:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(1)

    for x in Wins_2011:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(1)

    for x in Wins_2012:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(1)

    for x in Wins_2013:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(1)

    for x in Wins_2014:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(1)

    for x in Wins_2015:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(1)

    for x in Losses_1990:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(0)

    for x in Losses_1991:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(0)

    for x in Losses_1992:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(0)

    for x in Losses_1993:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(0)

    for x in Losses_1994:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(0)

    for x in Losses_1995:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(0)

    for x in Losses_1996:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(0)

    for x in Losses_1997:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(0)

    for x in Losses_1998:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(0)

    for x in Losses_1999:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(0)

    for x in Losses_2000:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(0)

    for x in Losses_2001:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(0)

    for x in Losses_2002:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(0)

    for x in Losses_2003:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(0)

    for x in Losses_2004:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(0)

    for x in Losses_2005:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(0)

    for x in Losses_2006:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(0)

    for x in Losses_2007:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(0)

    for x in Losses_2008:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(0)

    for x in Losses_2009:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(0)

    for x in Losses_2010:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(0)

    for x in Losses_2011:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(0)

    for x in Losses_2012:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(0)

    for x in Losses_2013:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(0)

    for x in Losses_2014:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(0)

    for x in Losses_2015:
        Game_Values.setdefault(x, [])
        Game_Values[x].append(0)

    return Game_Values

def Win_Probability(Game_Values):

    for x in Game_Values:
        Game_Values[x] = ((sum(Game_Values[x]))/(len(Game_Values[x])))
    return Game_Values

def sigmoid(x, x0, k):
     y = 1 / (1 + np.exp(-k*(x-x0)))
     return y

def Submission_Probabilities(Probability_Markers):
    Final_Probabilities = []
    for x in Probability_Markers:
        Probability_Markers = sigmoid(x, *popt)
        list.append(Final_Probabilities, Probability_Markers)
    return Final_Probabilities

def Prediction(IER_Dictionary, Submission_Format):

    Prediction_List = []
    for x in Submission_Format:

        Winner = x[1]
        Loser = x[2]

        Prediction = (1 / (1 + (10**(-(IER_Dictionary[Winner] - IER_Dictionary[Loser])/460))))
        list.append(Prediction_List, Prediction)
    return Prediction_List

def LL_Scoring(IM_Dictionary, Submission_Format):

    Prediction_List = []
    for x in Submission_Format:

        Winner = x[2]
        Loser = x[4]

        Prediction = (1 / (1 + (10**(-(IM_Dictionary[Winner] - IM_Dictionary[Loser])/460))))
        list.append(Prediction_List, Prediction)
    return Prediction_List

def M_Prediction(MasseyOrdinals, Submission_Format, X_Score):

    Prediction_List = []
    for x in Submission_Format:

        Winner = x[1]
        Loser = x[2]

        Prediction = (1 / (1 + (10**(-(MasseyOrdinals[Winner] - MasseyOrdinals[Loser])/X_Score))))
        list.append(Prediction_List, Prediction)
    return Prediction_List

def LogLoss(Actual, Predicted):

    Epsilon = 1e-15
    Predicted = sp.maximum(Epsilon, Predicted)

    Predicted = sp.minimum(1 - Epsilon, Predicted)
    LogLoss = sum(Actual * sp.log(Predicted) + sp.subtract(1, Actual) * sp.log(sp.subtract(1, Predicted)))
    LogLoss = LogLoss * -1.0 / len(Actual)
    return LogLoss

start = timeit.default_timer()

print ('-------------------------')
print ('Generating Arrays...', )

RegularSCR_1990 = np.asarray(Split_Array(RegularSCR, 1990))
RegularSCR_1991 = np.asarray(Split_Array(RegularSCR, 1991))
RegularSCR_1992 = np.asarray(Split_Array(RegularSCR, 1992))
RegularSCR_1993 = np.asarray(Split_Array(RegularSCR, 1993))
RegularSCR_1994 = np.asarray(Split_Array(RegularSCR, 1994))
RegularSCR_1995 = np.asarray(Split_Array(RegularSCR, 1995))
RegularSCR_1996 = np.asarray(Split_Array(RegularSCR, 1996))
RegularSCR_1997 = np.asarray(Split_Array(RegularSCR, 1997))
RegularSCR_1998 = np.asarray(Split_Array(RegularSCR, 1998))
RegularSCR_1999 = np.asarray(Split_Array(RegularSCR, 1999))
RegularSCR_2000 = np.asarray(Split_Array(RegularSCR, 2000))
RegularSCR_2001 = np.asarray(Split_Array(RegularSCR, 2001))

RegularSCR_2002 = np.asarray(Split_Array(RegularSCR, 2002))
TourneyCR_2002 = np.asarray(Split_Array(TourneyCR, 2002))
RegularSCR_2003 = np.asarray(Split_Array(RegularSCR, 2003))
TourneyCR_2003 = np.asarray(Split_Array(TourneyCR, 2003))
RegularSCR_2004 = np.asarray(Split_Array(RegularSCR, 2004))
TourneyCR_2004 = np.asarray(Split_Array(TourneyCR, 2004))
RegularSCR_2005 = np.asarray(Split_Array(RegularSCR, 2005))
TourneyCR_2005 = np.asarray(Split_Array(TourneyCR, 2005))
RegularSCR_2006 = np.asarray(Split_Array(RegularSCR, 2006))
TourneyCR_2006 = np.asarray(Split_Array(TourneyCR, 2006))
RegularSCR_2007 = np.asarray(Split_Array(RegularSCR, 2007))
TourneyCR_2007 = np.asarray(Split_Array(TourneyCR, 2007))
RegularSCR_2008 = np.asarray(Split_Array(RegularSCR, 2008))
TourneyCR_2008 = np.asarray(Split_Array(TourneyCR, 2008))
RegularSCR_2009 = np.asarray(Split_Array(RegularSCR, 2009))
TourneyCR_2009 = np.asarray(Split_Array(TourneyCR, 2009))
RegularSCR_2010 = np.asarray(Split_Array(RegularSCR, 2010))
TourneyCR_2010 = np.asarray(Split_Array(TourneyCR, 2010))
RegularSCR_2011 = np.asarray(Split_Array(RegularSCR, 2011))
TourneyCR_2011 = np.asarray(Split_Array(TourneyCR, 2011))
RegularSCR_2012 = np.asarray(Split_Array(RegularSCR, 2012))
TourneyCR_2012 = np.asarray(Split_Array(TourneyCR, 2012))
RegularSCR_2013 = np.asarray(Split_Array(RegularSCR, 2013))
TourneyCR_2013 = np.asarray(Split_Array(TourneyCR, 2013))
RegularSCR_2014 = np.asarray(Split_Array(RegularSCR, 2014))
TourneyCR_2014 = np.asarray(Split_Array(TourneyCR, 2014))
RegularSCR_2015 = np.asarray(Split_Array(RegularSCR, 2015))
TourneyCR_2015 = np.asarray(Split_Array(TourneyCR, 2015))
RegularSCR_2016 = np.asarray(Split_Array(RegularSCR, 2016))

print ('Optimizing K_Score...'),

K_Score_2002 = K_Score(TourneyCR_2002)
K_Score_2003 = K_Score(TourneyCR_2003)
K_Score_2004 = K_Score(TourneyCR_2004)
K_Score_2005 = K_Score(TourneyCR_2005)
K_Score_2006 = K_Score(TourneyCR_2006)
K_Score_2007 = K_Score(TourneyCR_2007)
K_Score_2008 = K_Score(TourneyCR_2008)
K_Score_2009 = K_Score(TourneyCR_2009)
K_Score_2010 = K_Score(TourneyCR_2010)
K_Score_2011 = K_Score(TourneyCR_2011)
K_Score_2012 = K_Score(TourneyCR_2012)
K_Score_2013 = K_Score(TourneyCR_2013)
K_Score_2014 = K_Score(TourneyCR_2014)
K_Score_2015 = K_Score(TourneyCR_2015)

print ('Creating Submission...'),

Submission_2012 = np.asarray(Split_Array(Submission, 2012))
Submission_2013 = np.asarray(Split_Array(Submission, 2013))
Submission_2014 = np.asarray(Split_Array(Submission, 2014))
Submission_2015 = np.asarray(Split_Array(Submission, 2015))
Submission_2016 = np.asarray(Split_Array(Submission_2016, 2016))

Teams_Array = np.asarray(Teams)
Teams_List = list(Teams_Array[:,0])

print ('-------------------------')

IER_Dictionary_Initial = Initial_Elo_Rating(Teams_List)

print ('Processing Elo Ratings (2002)...')
IER_Dictionary_Season_2002 = Elo_Ratings(IER_Dictionary_Initial, RegularSCR_2002)
IER_Dictionary_Tourney_2002 = Elo_Ratings_Tourney(IER_Dictionary_Season_2002, TourneyCR_2002, K_Score_2002)
IER_Dictionary_Depreciation_2002 = Elo_Rating_Depreciation(IER_Dictionary_Tourney_2002)

print ('Processing Elo Ratings (2003)...')
IER_Dictionary_Season_2003 = Elo_Ratings(IER_Dictionary_Depreciation_2002, RegularSCR_2003)
IER_Dictionary_Tourney_2003 = Elo_Ratings_Tourney(IER_Dictionary_Season_2003, TourneyCR_2003, K_Score_2003)
IER_Dictionary_Depreciation_2003 = Elo_Rating_Depreciation(IER_Dictionary_Tourney_2003)

print ('Processing Elo Ratings (2004)...')
IER_Dictionary_Season_2004 = Elo_Ratings(IER_Dictionary_Depreciation_2003, RegularSCR_2004)
IER_Dictionary_Tourney_2004 = Elo_Ratings_Tourney(IER_Dictionary_Season_2004, TourneyCR_2004, K_Score_2004)
IER_Dictionary_Depreciation_2004 = Elo_Rating_Depreciation(IER_Dictionary_Tourney_2004)

print ('Processing Elo Ratings (2005)...')
IER_Dictionary_Season_2005 = Elo_Ratings(IER_Dictionary_Depreciation_2004, RegularSCR_2005)
IER_Dictionary_Tourney_2005 = Elo_Ratings_Tourney(IER_Dictionary_Season_2005, TourneyCR_2005, K_Score_2005)
IER_Dictionary_Depreciation_2005 = Elo_Rating_Depreciation(IER_Dictionary_Tourney_2005)

print ('Processing Elo Ratings (2006)...')
IER_Dictionary_Season_2006 = Elo_Ratings(IER_Dictionary_Depreciation_2005, RegularSCR_2006)
IER_Dictionary_Tourney_2006 = Elo_Ratings_Tourney(IER_Dictionary_Season_2006, TourneyCR_2006, K_Score_2006)
IER_Dictionary_Depreciation_2006 = Elo_Rating_Depreciation(IER_Dictionary_Tourney_2006)

print ('Processing Elo Ratings (2007)...')
IER_Dictionary_Season_2007 = Elo_Ratings(IER_Dictionary_Depreciation_2006, RegularSCR_2007)
IER_Dictionary_Tourney_2007 = Elo_Ratings_Tourney(IER_Dictionary_Season_2007, TourneyCR_2007, K_Score_2007)
IER_Dictionary_Depreciation_2007 = Elo_Rating_Depreciation(IER_Dictionary_Tourney_2007)

print ('Processing Elo Ratings (2008)...')
IER_Dictionary_Season_2008 = Elo_Ratings(IER_Dictionary_Depreciation_2007, RegularSCR_2008)
IER_Dictionary_Tourney_2008 = Elo_Ratings_Tourney(IER_Dictionary_Season_2008, TourneyCR_2008, K_Score_2008)
IER_Dictionary_Depreciation_2008 = Elo_Rating_Depreciation(IER_Dictionary_Tourney_2008)

print ('Processing Elo Ratings (2009)...')
IER_Dictionary_Season_2009 = Elo_Ratings(IER_Dictionary_Depreciation_2008, RegularSCR_2009)
IER_Dictionary_Tourney_2009 = Elo_Ratings_Tourney(IER_Dictionary_Season_2009, TourneyCR_2009, K_Score_2009)
IER_Dictionary_Depreciation_2009 = Elo_Rating_Depreciation(IER_Dictionary_Tourney_2009)

print ('Processing Elo Ratings (2010)...')
IER_Dictionary_Season_2010 = Elo_Ratings(IER_Dictionary_Depreciation_2009, RegularSCR_2010)
IER_Dictionary_Tourney_2010 = Elo_Ratings_Tourney(IER_Dictionary_Season_2010, TourneyCR_2010, K_Score_2010)
IER_Dictionary_Depreciation_2010 = Elo_Rating_Depreciation(IER_Dictionary_Tourney_2010)

print ('Processing Elo Ratings (2011)...')
IER_Dictionary_Season_2011 = Elo_Ratings(IER_Dictionary_Depreciation_2010, RegularSCR_2011)
IER_Dictionary_Tourney_2011 = Elo_Ratings_Tourney(IER_Dictionary_Season_2011, TourneyCR_2011, K_Score_2011)
IER_Dictionary_Depreciation_2011 = Elo_Rating_Depreciation(IER_Dictionary_Tourney_2011)

print ('Processing Elo Ratings (2012)...')
IER_Dictionary_Season_2012 = Elo_Ratings(IER_Dictionary_Depreciation_2011, RegularSCR_2012)
Elo_Prediction_2012 = (Prediction(IER_Dictionary_Season_2012, Submission_2012))
Elo_LL_Prediction_2012 = LL_Scoring(IER_Dictionary_Season_2012, TourneyCR_2012)
IER_Dictionary_Tourney_2012 = Elo_Ratings_Tourney(IER_Dictionary_Season_2012, TourneyCR_2012, K_Score_2012)
IER_Dictionary_Depreciation_2012 = Elo_Rating_Depreciation(IER_Dictionary_Tourney_2012)

print ('Processing Elo Ratings (2013)...')
IER_Dictionary_Season_2013 = Elo_Ratings(IER_Dictionary_Depreciation_2012, RegularSCR_2013)
Elo_Prediction_2013 = (Prediction(IER_Dictionary_Season_2013, Submission_2013))
Elo_LL_Prediction_2013 = LL_Scoring(IER_Dictionary_Season_2013, TourneyCR_2013)
IER_Dictionary_Tourney_2013 = Elo_Ratings_Tourney(IER_Dictionary_Season_2012, TourneyCR_2013, K_Score_2013)
IER_Dictionary_Depreciation_2013 = Elo_Rating_Depreciation(IER_Dictionary_Tourney_2013)

print ('Processing Elo Ratings (2014)...')
IER_Dictionary_Season_2014 = Elo_Ratings(IER_Dictionary_Depreciation_2013, RegularSCR_2014)
Elo_Prediction_2014 = (Prediction(IER_Dictionary_Season_2014, Submission_2014))
Elo_LL_Prediction_2014 = LL_Scoring(IER_Dictionary_Season_2014, TourneyCR_2014)
IER_Dictionary_Tourney_2014 = Elo_Ratings_Tourney(IER_Dictionary_Season_2014, TourneyCR_2014, K_Score_2014)
IER_Dictionary_Depreciation_2014 = Elo_Rating_Depreciation(IER_Dictionary_Tourney_2014)

print ('Processing Elo Ratings (2015)...')
IER_Dictionary_Season_2015 = Elo_Ratings(IER_Dictionary_Depreciation_2014, RegularSCR_2015)
Elo_Prediction_2015 = (Prediction(IER_Dictionary_Season_2015, Submission_2015))
Elo_LL_Prediction_2015 = LL_Scoring(IER_Dictionary_Season_2015, TourneyCR_2015)
IER_Dictionary_Tourney_2015 = Elo_Ratings_Tourney(IER_Dictionary_Season_2015, TourneyCR_2015, K_Score_2015)
IER_Dictionary_Depreciation_2015 = Elo_Rating_Depreciation(IER_Dictionary_Tourney_2015)

print ('Processing Elo Ratings (2016)...')
IER_Dictionary_Season_2016 = Elo_Ratings(IER_Dictionary_Tourney_2015, RegularSCR_2016)
Elo_Prediction_2016 = (Prediction(IER_Dictionary_Season_2016, Submission_2016))

print ('-------------------------')

MasseyOrdinals_2012 = (np.asarray(list(zip(np.asarray(Split_Array(MasseyOrdinals, 2012))[:,3], np.asarray(Split_Array(MasseyOrdinals, 2012))[:,4]))))
MasseyOrdinals_2013 = (np.asarray(list(zip(np.asarray(Split_Array(MasseyOrdinals, 2013))[:,3], np.asarray(Split_Array(MasseyOrdinals, 2013))[:,4]))))
MasseyOrdinals_2014 = (np.asarray(list(zip(np.asarray(Split_Array(MasseyOrdinals, 2014))[:,3], np.asarray(Split_Array(MasseyOrdinals, 2014))[:,4]))))
MasseyOrdinals_2015 = (np.asarray(list(zip(np.asarray(Split_Array(MasseyOrdinals, 2015))[:,3], np.asarray(Split_Array(MasseyOrdinals, 2015))[:,4]))))
MasseyOrdinals_2016 = (np.asarray(list(zip(np.asarray(Split_Array(MasseyOrdinals_2016, 2016))[:,3], np.asarray(Split_Array(MasseyOrdinals_2016, 2016))[:,4]))))

MasseyOrdinals_2012_Dict = (Power_Rankings(MasseyOrdinals_2012, Teams_List))
print ('Processing Massey Ordinals (2012)...')
MasseyOrdinals_2013_Dict = (Power_Rankings(MasseyOrdinals_2013, Teams_List))
print ('Processing Massey Ordinals (2013)...')
MasseyOrdinals_2014_Dict = (Power_Rankings(MasseyOrdinals_2014, Teams_List))
print ('Processing Massey Ordinals (2014)...')
MasseyOrdinals_2015_Dict = (Power_Rankings(MasseyOrdinals_2015, Teams_List))
print ('Processing Massey Ordinals (2015)...')
MasseyOrdinals_2016_Dict = (Power_Rankings(MasseyOrdinals_2016, Teams_List))
print ('Processing Massey Ordinals (2016)...')

Massey_Prediction_2012 = (M_Prediction(MasseyOrdinals_2012_Dict, Submission_2012, 16.1161))
Massey_Prediction_2013 = (M_Prediction(MasseyOrdinals_2013_Dict, Submission_2013, 16.1161))
Massey_Prediction_2014 = (M_Prediction(MasseyOrdinals_2014_Dict, Submission_2014, 16.1161))
Massey_Prediction_2015 = (M_Prediction(MasseyOrdinals_2015_Dict, Submission_2015, 16.1161))
Massey_Prediction_2016 = (M_Prediction(MasseyOrdinals_2016_Dict, Submission_2016, 16.1161))

Massey_LL_Prediction_2012 = LL_Scoring(MasseyOrdinals_2012_Dict, TourneyCR_2012)
Massey_LL_Prediction_2013 = LL_Scoring(MasseyOrdinals_2013_Dict, TourneyCR_2013)
Massey_LL_Prediction_2014 = LL_Scoring(MasseyOrdinals_2014_Dict, TourneyCR_2014)
Massey_LL_Prediction_2015 = LL_Scoring(MasseyOrdinals_2015_Dict, TourneyCR_2015)

print ('-------------------------')

Games_1990 = GameData(RegularSCR_1990)
Games_1991 = GameData(RegularSCR_1991)
Games_1992 = GameData(RegularSCR_1992)
Games_1993 = GameData(RegularSCR_1993)
Games_1994 = GameData(RegularSCR_1994)
Games_1995 = GameData(RegularSCR_1995)
Games_1996 = GameData(RegularSCR_1996)
Games_1997 = GameData(RegularSCR_1997)
Games_1998 = GameData(RegularSCR_1998)
Games_1999 = GameData(RegularSCR_1999)
Games_2000 = GameData(RegularSCR_2000)
Games_2001 = GameData(RegularSCR_2001)
Games_2002 = GameData(RegularSCR_2002)
Games_2003 = GameData(RegularSCR_2003)
Games_2004 = GameData(RegularSCR_2004)
Games_2005 = GameData(RegularSCR_2005)
Games_2006 = GameData(RegularSCR_2006)
Games_2007 = GameData(RegularSCR_2007)
Games_2008 = GameData(RegularSCR_2008)
Games_2009 = GameData(RegularSCR_2009)
Games_2010 = GameData(RegularSCR_2010)
Games_2011 = GameData(RegularSCR_2011)
Games_2012 = GameData(RegularSCR_2012)
Games_2013 = GameData(RegularSCR_2013)
Games_2014 = GameData(RegularSCR_2014)
Games_2015 = GameData(RegularSCR_2015)
Games_2016 = GameData(RegularSCR_2016)

Games_Matrix_1990 = BuildGames(Games_1990)
Games_Matrix_1991 = BuildGames(Games_1991)
Games_Matrix_1992 = BuildGames(Games_1992)
Games_Matrix_1993 = BuildGames(Games_1993)
Games_Matrix_1994 = BuildGames(Games_1994)
Games_Matrix_1995 = BuildGames(Games_1995)
Games_Matrix_1996 = BuildGames(Games_1996)
Games_Matrix_1997 = BuildGames(Games_1997)
Games_Matrix_1998 = BuildGames(Games_1998)
Games_Matrix_1999 = BuildGames(Games_1999)
Games_Matrix_2000 = BuildGames(Games_2000)
Games_Matrix_2001 = BuildGames(Games_2001)
Games_Matrix_2002 = BuildGames(Games_2002)
Games_Matrix_2003 = BuildGames(Games_2003)
Games_Matrix_2004 = BuildGames(Games_2004)
Games_Matrix_2005 = BuildGames(Games_2005)
Games_Matrix_2006 = BuildGames(Games_2006)
Games_Matrix_2007 = BuildGames(Games_2007)
Games_Matrix_2008 = BuildGames(Games_2008)
Games_Matrix_2009 = BuildGames(Games_2009)
Games_Matrix_2010 = BuildGames(Games_2010)
Games_Matrix_2011 = BuildGames(Games_2011)
Games_Matrix_2012 = BuildGames(Games_2012)
Games_Matrix_2013 = BuildGames(Games_2013)
Games_Matrix_2014 = BuildGames(Games_2014)
Games_Matrix_2015 = BuildGames(Games_2015)
Games_Matrix_2016 = BuildGames(Games_2016)

Games_Outcome_1990 = BuildOutcomes(Games_1990)
Games_Outcome_1991 = BuildOutcomes(Games_1991)
Games_Outcome_1992 = BuildOutcomes(Games_1992)
Games_Outcome_1993 = BuildOutcomes(Games_1993)
Games_Outcome_1994 = BuildOutcomes(Games_1994)
Games_Outcome_1995 = BuildOutcomes(Games_1995)
Games_Outcome_1996 = BuildOutcomes(Games_1996)
Games_Outcome_1997 = BuildOutcomes(Games_1997)
Games_Outcome_1998 = BuildOutcomes(Games_1998)
Games_Outcome_1999 = BuildOutcomes(Games_1999)
Games_Outcome_2000 = BuildOutcomes(Games_2000)
Games_Outcome_2001 = BuildOutcomes(Games_2001)
Games_Outcome_2002 = BuildOutcomes(Games_2002)
Games_Outcome_2003 = BuildOutcomes(Games_2003)
Games_Outcome_2004 = BuildOutcomes(Games_2004)
Games_Outcome_2005 = BuildOutcomes(Games_2005)
Games_Outcome_2006 = BuildOutcomes(Games_2006)
Games_Outcome_2007 = BuildOutcomes(Games_2007)
Games_Outcome_2008 = BuildOutcomes(Games_2008)
Games_Outcome_2009 = BuildOutcomes(Games_2009)
Games_Outcome_2010 = BuildOutcomes(Games_2010)
Games_Outcome_2011 = BuildOutcomes(Games_2011)
Games_Outcome_2012 = BuildOutcomes(Games_2012)
Games_Outcome_2013 = BuildOutcomes(Games_2013)
Games_Outcome_2014 = BuildOutcomes(Games_2014)
Games_Outcome_2015 = BuildOutcomes(Games_2015)
Games_Outcome_2016 = BuildOutcomes(Games_2016)

Games_LS_1990 = np.linalg.lstsq(Games_Matrix_1990, Games_Outcome_1990)[0]
Games_LS_1991 = np.linalg.lstsq(Games_Matrix_1991, Games_Outcome_1991)[0]
Games_LS_1992 = np.linalg.lstsq(Games_Matrix_1992, Games_Outcome_1992)[0]
Games_LS_1993 = np.linalg.lstsq(Games_Matrix_1993, Games_Outcome_1993)[0]
Games_LS_1994 = np.linalg.lstsq(Games_Matrix_1994, Games_Outcome_1994)[0]
Games_LS_1995 = np.linalg.lstsq(Games_Matrix_1995, Games_Outcome_1995)[0]
Games_LS_1996 = np.linalg.lstsq(Games_Matrix_1996, Games_Outcome_1996)[0]
Games_LS_1997 = np.linalg.lstsq(Games_Matrix_1997, Games_Outcome_1997)[0]
Games_LS_1998 = np.linalg.lstsq(Games_Matrix_1998, Games_Outcome_1998)[0]
Games_LS_1999 = np.linalg.lstsq(Games_Matrix_1999, Games_Outcome_1999)[0]
Games_LS_2000 = np.linalg.lstsq(Games_Matrix_2000, Games_Outcome_2000)[0]
Games_LS_2001 = np.linalg.lstsq(Games_Matrix_2001, Games_Outcome_2001)[0]
Games_LS_2002 = np.linalg.lstsq(Games_Matrix_2002, Games_Outcome_2002)[0]
Games_LS_2003 = np.linalg.lstsq(Games_Matrix_2003, Games_Outcome_2003)[0]
Games_LS_2004 = np.linalg.lstsq(Games_Matrix_2004, Games_Outcome_2004)[0]
Games_LS_2005 = np.linalg.lstsq(Games_Matrix_2005, Games_Outcome_2005)[0]
Games_LS_2006 = np.linalg.lstsq(Games_Matrix_2006, Games_Outcome_2006)[0]
Games_LS_2007 = np.linalg.lstsq(Games_Matrix_2007, Games_Outcome_2007)[0]
Games_LS_2008 = np.linalg.lstsq(Games_Matrix_2008, Games_Outcome_2008)[0]
Games_LS_2009 = np.linalg.lstsq(Games_Matrix_2009, Games_Outcome_2009)[0]
Games_LS_2010 = np.linalg.lstsq(Games_Matrix_2010, Games_Outcome_2010)[0]
Games_LS_2011 = np.linalg.lstsq(Games_Matrix_2011, Games_Outcome_2011)[0]
Games_LS_2012 = np.linalg.lstsq(Games_Matrix_2012, Games_Outcome_2012)[0]
Games_LS_2013 = np.linalg.lstsq(Games_Matrix_2013, Games_Outcome_2013)[0]
Games_LS_2014 = np.linalg.lstsq(Games_Matrix_2014, Games_Outcome_2014)[0]
Games_LS_2015 = np.linalg.lstsq(Games_Matrix_2015, Games_Outcome_2015)[0]
Games_LS_2016 = np.linalg.lstsq(Games_Matrix_2016, Games_Outcome_2016)[0]

print ('Processing Regression Data (1990)...')
Probabilities_1990 = Probabilities(RegularSCR_1990, Games_1990, RegularSCR_1990, Games_LS_1990)
print ('Processing Regression Data (1991)...')
Probabilities_1991 = Probabilities(RegularSCR_1991, Games_1991, RegularSCR_1991, Games_LS_1991)
print ('Processing Regression Data (1992)...')
Probabilities_1992 = Probabilities(RegularSCR_1992, Games_1992, RegularSCR_1992, Games_LS_1992)
print ('Processing Regression Data (1993)...')
Probabilities_1993 = Probabilities(RegularSCR_1993, Games_1993, RegularSCR_1993, Games_LS_1993)
print ('Processing Regression Data (1994)...')
Probabilities_1994 = Probabilities(RegularSCR_1994, Games_1994, RegularSCR_1994, Games_LS_1994)
print ('Processing Regression Data (1995)...')
Probabilities_1995 = Probabilities(RegularSCR_1995, Games_1995, RegularSCR_1995, Games_LS_1995)
print ('Processing Regression Data (1996)...')
Probabilities_1996 = Probabilities(RegularSCR_1996, Games_1996, RegularSCR_1996, Games_LS_1996)
print ('Processing Regression Data (1997)...')
Probabilities_1997 = Probabilities(RegularSCR_1997, Games_1997, RegularSCR_1997, Games_LS_1997)
print ('Processing Regression Data (1998)...')
Probabilities_1998 = Probabilities(RegularSCR_1998, Games_1998, RegularSCR_1998, Games_LS_1998)
print ('Processing Regression Data (1999)...')
Probabilities_1999 = Probabilities(RegularSCR_1999, Games_1999, RegularSCR_1999, Games_LS_1999)
print ('Processing Regression Data (2000)...')
Probabilities_2000 = Probabilities(RegularSCR_2000, Games_2000, RegularSCR_2000, Games_LS_2000)
print ('Processing Regression Data (2001)...')
Probabilities_2001 = Probabilities(RegularSCR_2001, Games_2001, RegularSCR_2001, Games_LS_2001)
print ('Processing Regression Data (2002)...')
Probabilities_2002 = Probabilities(RegularSCR_2002, Games_2002, RegularSCR_2002, Games_LS_2002)
print ('Processing Regression Data (2003)...')
Probabilities_2003 = Probabilities(RegularSCR_2003, Games_2003, RegularSCR_2003, Games_LS_2003)
print ('Processing Regression Data (2004)...')
Probabilities_2004 = Probabilities(RegularSCR_2004, Games_2004, RegularSCR_2004, Games_LS_2004)
print ('Processing Regression Data (2005)...')
Probabilities_2005 = Probabilities(RegularSCR_2005, Games_2005, RegularSCR_2005, Games_LS_2005)
print ('Processing Regression Data (2006)...')
Probabilities_2006 = Probabilities(RegularSCR_2006, Games_2006, RegularSCR_2006, Games_LS_2006)
print ('Processing Regression Data (2007)...')
Probabilities_2007 = Probabilities(RegularSCR_2007, Games_2007, RegularSCR_2007, Games_LS_2007)
print ('Processing Regression Data (2008)...')
Probabilities_2008 = Probabilities(RegularSCR_2008, Games_2008, RegularSCR_2008, Games_LS_2008)
print ('Processing Regression Data (2009)...')
Probabilities_2009 = Probabilities(RegularSCR_2009, Games_2009, RegularSCR_2009, Games_LS_2009)
print ('Processing Regression Data (2010)...')
Probabilities_2010 = Probabilities(RegularSCR_2010, Games_2010, RegularSCR_2010, Games_LS_2010)
print ('Processing Regression Data (2011)...')
Probabilities_2011 = Probabilities(RegularSCR_2011, Games_2011, RegularSCR_2011, Games_LS_2011)
print ('Processing Regression Data (2012)...')
Probabilities_2012 = Probabilities(RegularSCR_2012, Games_2012, RegularSCR_2012, Games_LS_2012)
print ('Processing Regression Data (2013)...')
Probabilities_2013 = Probabilities(RegularSCR_2013, Games_2013, RegularSCR_2013, Games_LS_2013)
print ('Processing Regression Data (2014)...')
Probabilities_2014 = Probabilities(RegularSCR_2014, Games_2014, RegularSCR_2014, Games_LS_2014)
print ('Processing Regression Data (2015)...')
Probabilities_2015 = Probabilities(RegularSCR_2015, Games_2015, RegularSCR_2015, Games_LS_2015)
print ('Processing Regression Data (2016)...')
Probabilities_2016 = Probabilities(RegularSCR_2016, Games_2016, RegularSCR_2016, Games_LS_2016)

Regression_Prediction_2012 = TProbabilities(Submission_2012, Games_2012, RegularSCR_2012, Games_LS_2012)
Regression_Prediction_2013 = TProbabilities(Submission_2013, Games_2013, RegularSCR_2013, Games_LS_2013)
Regression_Prediction_2014 = TProbabilities(Submission_2014, Games_2014, RegularSCR_2014, Games_LS_2014)
Regression_Prediction_2015 = TProbabilities(Submission_2015, Games_2015, RegularSCR_2015, Games_LS_2015)

Regression_Prediction_2016 = TProbabilities(Submission_2016, Games_2016, RegularSCR_2016, Games_LS_2016)

Regression_LL_Prediction_2012 = Probabilities(TourneyCR_2012, Games_2012, RegularSCR_2012, Games_LS_2012)
Regression_LL_Prediction_2013 = Probabilities(TourneyCR_2013, Games_2013, RegularSCR_2013, Games_LS_2013)
Regression_LL_Prediction_2014 = Probabilities(TourneyCR_2014, Games_2014, RegularSCR_2014, Games_LS_2014)
Regression_LL_Prediction_2015 = Probabilities(TourneyCR_2015, Games_2015, RegularSCR_2015, Games_LS_2015)

Wins_1990 = map(lambda x: Probabilities_1990[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_1990))))
List_1990 = map(lambda x: Probabilities_1990[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_1990))))
Losses_1990 = [-x for x in List_1990]

Wins_1991 = map(lambda x: Probabilities_1991[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_1991))))
List_1991 = map(lambda x: Probabilities_1991[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_1991))))
Losses_1991 = [-x for x in List_1991]

Wins_1992 = map(lambda x: Probabilities_1992[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_1992))))
List_1992 = map(lambda x: Probabilities_1992[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_1992))))
Losses_1992 = [-x for x in List_1992]

Wins_1993 = map(lambda x: Probabilities_1993[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_1993))))
List_1993 = map(lambda x: Probabilities_1993[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_1993))))
Losses_1993 = [-x for x in List_1993]

Wins_1994 = map(lambda x: Probabilities_1994[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_1994))))
List_1994 = map(lambda x: Probabilities_1994[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_1994))))
Losses_1994 = [-x for x in List_1994]

Wins_1995 = map(lambda x: Probabilities_1995[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_1995))))
List_1995 = map(lambda x: Probabilities_1995[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_1995))))
Losses_1995 = [-x for x in List_1995]

Wins_1996 = map(lambda x: Probabilities_1996[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_1996))))
List_1996 = map(lambda x: Probabilities_1996[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_1996))))
Losses_1996 = [-x for x in List_1996]

Wins_1997 = map(lambda x: Probabilities_1997[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_1997))))
List_1997 = map(lambda x: Probabilities_1997[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_1997))))
Losses_1997 = [-x for x in List_1997]

Wins_1998 = map(lambda x: Probabilities_1998[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_1998))))
List_1998 = map(lambda x: Probabilities_1998[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_1998))))
Losses_1998 = [-x for x in List_1998]

Wins_1999 = map(lambda x: Probabilities_1999[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_1999))))
List_1999 = map(lambda x: Probabilities_1999[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_1999))))
Losses_1999 = [-x for x in List_1999]

Wins_2000 = map(lambda x: Probabilities_2000[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_2000))))
List_2000 = map(lambda x: Probabilities_2000[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_2000))))
Losses_2000 = [-x for x in List_2000]

Wins_2001 = map(lambda x: Probabilities_2001[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_2001))))
List_2001 = map(lambda x: Probabilities_2001[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_2001))))
Losses_2001 = [-x for x in List_2001]

Wins_2002 = map(lambda x: Probabilities_2002[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_2002))))
List_2002 = map(lambda x: Probabilities_2002[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_2002))))
Losses_2002 = [-x for x in List_2002]

Wins_2003 = map(lambda x: Probabilities_2003[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_2003))))
List_2003 = map(lambda x: Probabilities_2003[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_2003))))
Losses_2003 = [-x for x in List_2003]

Wins_2004 = map(lambda x: Probabilities_2004[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_2004))))
List_2004 = map(lambda x: Probabilities_2004[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_2004))))
Losses_2004 = [-x for x in List_2004]

Wins_2005 = map(lambda x: Probabilities_2005[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_2005))))
List_2005 = map(lambda x: Probabilities_2005[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_2005))))
Losses_2005 = [-x for x in List_2005]

Wins_2006 = map(lambda x: Probabilities_2006[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_2006))))
List_2006 = map(lambda x: Probabilities_2006[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_2006))))
Losses_2006 = [-x for x in List_2006]

Wins_2007 = map(lambda x: Probabilities_2007[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_2007))))
List_2007 = map(lambda x: Probabilities_2007[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_2007))))
Losses_2007 = [-x for x in List_2007]

Wins_2008 = map(lambda x: Probabilities_2008[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_2008))))
List_2008 = map(lambda x: Probabilities_2008[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_2008))))
Losses_2008 = [-x for x in List_2008]

Wins_2009 = map(lambda x: Probabilities_2009[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_2009))))
List_2009 = map(lambda x: Probabilities_2009[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_2009))))
Losses_2009 = [-x for x in List_2009]

Wins_2010 = map(lambda x: Probabilities_2010[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_2010))))
List_2010 = map(lambda x: Probabilities_2010[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_2010))))
Losses_2010 = [-x for x in List_2010]

Wins_2011 = map(lambda x: Probabilities_2011[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_2011))))
List_2011 = map(lambda x: Probabilities_2011[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_2011))))
Losses_2011 = [-x for x in List_2011]

Wins_2012 = map(lambda x: Probabilities_2012[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_2012))))
List_2012 = map(lambda x: Probabilities_2012[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_2012))))
Losses_2012 = [-x for x in List_2012]

Wins_2013 = map(lambda x: Probabilities_2013[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_2013))))
List_2013 = map(lambda x: Probabilities_2013[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_2013))))
Losses_2013 = [-x for x in List_2013]

Wins_2014 = map(lambda x: Probabilities_2014[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_2014))))
List_2014 = map(lambda x: Probabilities_2014[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_2014))))
Losses_2014 = [-x for x in List_2014]

Wins_2015 = map(lambda x: Probabilities_2015[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_2015))))
List_2015 = map(lambda x: Probabilities_2015[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_2015))))
Losses_2015 = [-x for x in List_2015]

Wins_2016 = map(lambda x: Probabilities_2016[x], filter(lambda x: x % 2 == 0, range(len(Probabilities_2016))))
List_2016 = map(lambda x: Probabilities_2016[x], filter(lambda x: x % 2 == 1, range(len(Probabilities_2016))))
Losses_2016 = [-x for x in List_2016]

GameValues_Total = GameValues(Wins_1990, Wins_1991, Wins_1992, Wins_1993, Wins_1994, Wins_1995, Wins_1996, Wins_1997,
                              Wins_1998, Wins_1999, Wins_2000, Wins_2001, Wins_2002, Wins_2003, Wins_2004, Wins_2005,
                              Wins_2006, Wins_2007, Wins_2008, Wins_2009, Wins_2010, Wins_2011, Wins_2012, Wins_2013,
                              Wins_2014, Wins_2015, Losses_1990, Losses_1991, Losses_1992, Losses_1993, Losses_1994,
                              Losses_1995, Losses_1996, Losses_1997, Losses_1998, Losses_1999, Losses_2000, Losses_2001,
                              Losses_2002, Losses_2003, Losses_2004, Losses_2005, Losses_2006, Losses_2007, Losses_2008,
                              Losses_2009, Losses_2010, Losses_2011, Losses_2012, Losses_2013, Losses_2014, Losses_2015)

WinProbability_Total = collections.OrderedDict(sorted(Win_Probability(GameValues_Total).items(), key=operator.itemgetter(0)))

xdata = list((WinProbability_Total.keys()))
ydata = list((WinProbability_Total.values()))
popt, pcov = curve_fit(sigmoid, xdata, ydata)

print ('-------------------------')
print ('Optimizing Model...')

Elo_LL_Prediction_Markers = Elo_LL_Prediction_2012 + Elo_LL_Prediction_2013 + Elo_LL_Prediction_2014 + Elo_LL_Prediction_2015
Massey_LL_Prediction_Markers = Massey_LL_Prediction_2012 + Massey_LL_Prediction_2013 + Massey_LL_Prediction_2014 + Massey_LL_Prediction_2015
Regression_LL_Prediction_Markers = Regression_LL_Prediction_2012 + Regression_LL_Prediction_2013 + Regression_LL_Prediction_2014 + Regression_LL_Prediction_2015
Regression_LL_Prediction_Markers = Submission_Probabilities(Regression_LL_Prediction_Markers)

Elo_Prediction_Markers = Elo_Prediction_2012 + Elo_Prediction_2013 + Elo_Prediction_2014 + Elo_Prediction_2015
Massey_Prediction_Markers = Massey_Prediction_2012 + Massey_Prediction_2013 + Massey_Prediction_2014 + Massey_Prediction_2015
Regression_Prediction_Markers = Regression_Prediction_2012 + Regression_Prediction_2013 + Regression_Prediction_2014 + Regression_Prediction_2015
Regression_Prediction_Markers = Submission_Probabilities(Regression_Prediction_Markers)

Elo_Submission = np.asarray(Final_Submission)
Ordinals_Submission = np.asarray(Final_Submission)
Regression_Submission = np.asarray(Final_Submission)

Elo_Submission_2016 = np.asarray(Final_Submission_2016)
Ordinals_Submission_2016 = np.asarray(Final_Submission_2016)
Regression_Submission_2016 = np.asarray(Final_Submission_2016)

Elo_Submission[:,1] = Elo_Prediction_Markers
Ordinals_Submission[:,1] = Massey_Prediction_Markers
Regression_Submission[:,1] = Regression_Prediction_Markers

with open('C:/Users/Nick/Desktop/Massey Predictor/Final_Submissions/Final Predictions (Elo 2012-2015).csv',newline='') as f:
    r = csv.reader(f)
    data = [line for line in r]

with open('C:/Users/Nick/Desktop/Massey Predictor/Final_Submissions/Final Predictions (Elo 2012-2015).csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['ID','Pred'])
    w.writerows(Elo_Submission)

with open('C:/Users/Nick/Desktop/Massey Predictor/Final_Submissions/Final Predictions (Ordinal 2012-2015).csv',newline='') as f:
    r = csv.reader(f)
    data = [line for line in r]

with open('C:/Users/Nick/Desktop/Massey Predictor/Final_Submissions/Final Predictions (Ordinal 2012-2015).csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['ID','Pred'])
    w.writerows(Ordinals_Submission)

with open('C:/Users/Nick/Desktop/Massey Predictor/Final_Submissions/Final Predictions (Regression 2012-2015).csv',newline='') as f:
    r = csv.reader(f)
    data = [line for line in r]

with open('C:/Users/Nick/Desktop/Massey Predictor/Final_Submissions/Final Predictions (Regression 2012-2015).csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['ID','Pred'])
    w.writerows(Regression_Submission)

Elo_Prediction_Markers_2016 = Elo_Prediction_2016
Massey_Prediction_Markers_2016 = Massey_Prediction_2016
Regression_Prediction_Markers_2016 = Regression_Prediction_2016
Regression_Prediction_Markers_2016 = Submission_Probabilities(Regression_Prediction_Markers_2016)

Elo_Submission_2016[:,1] = Elo_Prediction_Markers_2016
Ordinals_Submission_2016[:,1] = Massey_Prediction_Markers_2016
Regression_Submission_2016[:,1] = Regression_Prediction_Markers_2016

with open('C:/Users/Nick/Desktop/Massey Predictor/Final_Submissions/Final Predictions (Elo 2016).csv',newline='') as f:
    r = csv.reader(f)
    data = [line for line in r]

with open('C:/Users/Nick/Desktop/Massey Predictor/Final_Submissions/Final Predictions (Elo 2016).csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['ID','Pred'])
    w.writerows(Elo_Submission_2016)

with open('C:/Users/Nick/Desktop/Massey Predictor/Final_Submissions/Final Predictions (Ordinal 2016).csv',newline='') as f:
    r = csv.reader(f)
    data = [line for line in r]

with open('C:/Users/Nick/Desktop/Massey Predictor/Final_Submissions/Final Predictions (Ordinal 2016).csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['ID','Pred'])
    w.writerows(Ordinals_Submission_2016)

with open('C:/Users/Nick/Desktop/Massey Predictor/Final_Submissions/Final Predictions (Regression 2016).csv',newline='') as f:
    r = csv.reader(f)
    data = [line for line in r]

with open('C:/Users/Nick/Desktop/Massey Predictor/Final_Submissions/Final Predictions (Regression 2016).csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['ID','Pred'])
    w.writerows(Regression_Submission_2016)

Net_Prophet_Ratings_Array = np.asarray(NetProphet_Ratings)
Net_Prophet_Ratings_Array_2016 = np.asarray(NetProphet_Ratings_2016)
Net_Prophet_Predictions = list(Net_Prophet_Ratings_Array[:,1])
Net_Prophet_Predictions_2016 = list(Net_Prophet_Ratings_Array_2016[:,1])

Final_Predictions = [(((0.123788522 * x) + (y * (1 - 0.123788522)))) for x, y in zip(Elo_Prediction_Markers, Massey_Prediction_Markers)]
Final_Predictions = [(((0.663466086 * x) + (y * (1 - 0.663466086)))) for x, y in zip(Final_Predictions, Regression_Prediction_Markers)]
Final_Predictions = np.clip(Final_Predictions, .0001, .9999)
Ensemble_Predictions = [(((0.005726196 * x) + (y * (1 - 0.005726196)))) for x, y in zip(Final_Predictions, Net_Prophet_Predictions)]
Ensemble_Predictions = np.clip(Ensemble_Predictions, .0001, .9999)

Final_Predictions_2016 = [(((0.123788522 * x) + (y * (1 - 0.123788522)))) for x, y in zip(Elo_Prediction_Markers_2016, Massey_Prediction_Markers_2016)]
Final_Predictions_2016 = [(((0.663466086 * x) + (y * (1 - 0.663466086)))) for x, y in zip(Final_Predictions_2016, Regression_Prediction_Markers_2016)]
Final_Predictions_2016 = np.clip(Final_Predictions_2016, .0001, .9999)
Ensemble_Predictions_2016 = [(((0.005726196 * x) + (y * (1 - 0.005726196)))) for x, y in zip(Final_Predictions_2016, Net_Prophet_Predictions_2016)]
Ensemble_Predictions_2016 = np.clip(Ensemble_Predictions_2016, .0001, .9999)

Final_Predictions_LL = [(((0.123788522 * x) + (y * (1 - 0.123788522)))) for x, y in zip(Elo_LL_Prediction_Markers, Massey_LL_Prediction_Markers)]
Final_Predictions_LL = [(((0.663466086 * x) + (y * (1 - 0.663466086)))) for x, y in zip(Final_Predictions_LL, Regression_LL_Prediction_Markers)]
Final_Predictions_LL = np.clip(Final_Predictions_LL, .0001, .9999)

LL_Actual = ([1] * len(Final_Predictions_LL))
LL_Predicted = Final_Predictions_LL
LL_Score = LogLoss(LL_Actual, LL_Predicted)

Submission_Markers = np.asarray(Final_Submission)
Submission_Markers_2016 = np.asarray(Final_Submission_2016)
Submission_Markers[:,1] = Final_Predictions
Submission_Markers_2016[:,1] = Final_Predictions_2016

Ensemble_Markers = np.asarray(Final_Submission)
Ensemble_Markers_2016 = np.asarray(Final_Submission_2016)
Ensemble_Markers[:,1] = Ensemble_Predictions
Ensemble_Markers_2016[:,1] = Ensemble_Predictions_2016

print ('Writing to CSV...')

with open('C:/Users/Nick/Desktop/Massey Predictor/Final_Submissions/Final Predictions (Model 2012-2015).csv',newline='') as f:
    r = csv.reader(f)
    data = [line for line in r]

with open('C:/Users/Nick/Desktop/Massey Predictor/Final_Submissions/Final Predictions (Model 2012-2015).csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['ID','Pred'])
    w.writerows(Submission_Markers)

with open('C:/Users/Nick/Desktop/Massey Predictor/Final_Submissions/Final Predictions (Model 2016).csv',newline='') as f:
    r = csv.reader(f)
    data = [line for line in r]

with open('C:/Users/Nick/Desktop/Massey Predictor/Final_Submissions/Final Predictions (Model 2016).csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['ID','Pred'])
    w.writerows(Submission_Markers_2016)

with open('C:/Users/Nick/Desktop/Massey Predictor/Final_Submissions/Final Predictions (Ensemble 2012-2015).csv',newline='') as f:
    r = csv.reader(f)
    data = [line for line in r]

with open('C:/Users/Nick/Desktop/Massey Predictor/Final_Submissions/Final Predictions (Ensemble 2012-2015).csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['ID','Pred'])
    w.writerows(Ensemble_Markers)

with open('C:/Users/Nick/Desktop/Massey Predictor/Final_Submissions/Final Predictions (Ensemble 2016).csv',newline='') as f:
    r = csv.reader(f)
    data = [line for line in r]

with open('C:/Users/Nick/Desktop/Massey Predictor/Final_Submissions/Final Predictions (Ensemble 2016).csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['ID','Pred'])
    w.writerows(Ensemble_Markers_2016)

print ('-------------------------')

stop = timeit.default_timer()

print ('Massey_Model_Final.py Completed')
print ('Model Score:', (-1 * (np.log(LL_Score))))
print ('Function Runtime:', round((stop - start) , 5),'s.')
print ('-------------------------')