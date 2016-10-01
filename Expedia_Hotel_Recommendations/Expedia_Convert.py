import pandas as pd
import os

file_to_convert = pd.read_csv(os.path.expanduser('~/Desktop/Expedia/Submission/submission.csv'))
file_to_convert['id'] = file_to_convert.index

for col in file_to_convert.columns:
    if 'Unnamed' in col:
        del file_to_convert[col]

print (file_to_convert)

file_to_convert.to_csv(os.path.expanduser('~/Desktop/Expedia/Submission/submission.csv'), index=False)