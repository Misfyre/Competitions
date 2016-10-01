__author__ = 'Nick Sarris'

import numpy as np
import pandas as pd

Predictions_Ensemble = pd.read_csv('~/Desktop/Massey Predictor/Final_Submissions/Final Predictions (Ensemble 2016).csv', encoding = "ISO-8859-1")
Predictions_Model = pd.read_csv('~/Desktop/Massey Predictor/Final_Submissions/Final Predictions (Model 2016).csv', encoding = "ISO-8859-1")
Bracket_Teams = pd.read_csv('~/Desktop/Massey Predictor/Bracket_Data/Bracket Teams.csv', encoding = "ISO-8859-1")

Teams_Array = np.asarray(Bracket_Teams)
Teams_List_1 = list(Teams_Array[:,0])
Teams_List_2 = list(Teams_Array[:,1])

Team_Dictionary = dict(zip(Teams_List_1, Teams_List_2))
ID_Dictionary = dict(zip(Teams_List_2, Teams_List_1))

Predictions_Model = np.asarray(Predictions_Model)
Predictions_Model_Keys = Predictions_Model[:,0]
Predictions_Model_Values = Predictions_Model[:,1]
Predictions_Model = dict(zip(Predictions_Model_Keys, Predictions_Model_Values))

Predictions_Ensemble = np.asarray(Predictions_Ensemble)
Predictions_Ensemble_Keys = Predictions_Ensemble[:,0]
Predictions_Ensemble_Values = Predictions_Ensemble[:,1]
Predictions_Ensemble = dict(zip(Predictions_Ensemble_Keys, Predictions_Ensemble_Values))

def Predictions(Team_Dictionary, ID_Dictionary, Predictions_Model, Predictions_Ensemble):
    
    Team_1 = ID_Dictionary[input("Enter Team 1: ")]
    Team_2 = ID_Dictionary[input("Enter Team 2: ")]

    print ('')

    try:
        Match_1 = ('2016_%d_%d' % (Team_1, Team_2))
        print ('Model Prediction = %s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Team_1], (100 * (Predictions_Model[Match_1])), Team_Dictionary[Team_2], (100 * (1 - Predictions_Model[Match_1]))))
               
    except:
        Match_1 = ('2016_%d_%d' % (Team_2, Team_1))
        print ('Model Prediction = %s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Team_2], (100 * (Predictions_Model[Match_1])), Team_Dictionary[Team_1], (100 * (1 - Predictions_Model[Match_1]))))

    try:
        Match_1 = ('2016_%d_%d' % (Team_1, Team_2))
        print ('Ensemble Prediction = %s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Team_1], (100 * (Predictions_Ensemble[Match_1])), Team_Dictionary[Team_2], (100 * (1 - Predictions_Ensemble[Match_1]))))
               
    except:
        Match_1 = ('2016_%d_%d' % (Team_2, Team_1))
        print ('Ensemble Prediction = %s (%.1f%%) vs %s (%.1f%%).' % (Team_Dictionary[Team_2], (100 * (Predictions_Ensemble[Match_1])), Team_Dictionary[Team_1], (100 * (1 - Predictions_Ensemble[Match_1]))))

Predictions(Team_Dictionary, ID_Dictionary, Predictions_Model, Predictions_Ensemble)