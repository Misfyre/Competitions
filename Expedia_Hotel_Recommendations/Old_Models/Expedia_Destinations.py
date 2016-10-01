import pandas as pd
import os

def probability_converter(x):
    return 10**x

destinations = pd.read_csv(os.path.expanduser('~/Desktop/Expedia/Data/destinations.csv'))
cols = [col for col in destinations.columns if col not in ['srch_destination_id']]
f_destinations = destinations[cols].apply(probability_converter, axis=1)

g_destinations = f_destinations.idxmax(axis=1)
final_probabilities = pd.concat([destinations['srch_destination_id'], g_destinations], axis=1)
final_probabilities.to_csv(os.path.expanduser('~/Desktop/Expedia/Data/destination_probabilities.csv'), index=False)