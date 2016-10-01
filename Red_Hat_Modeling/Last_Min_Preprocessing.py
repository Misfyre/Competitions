__author__ = "Nick Sarris (ngs5st)"

import pandas as pd
submission = pd.read_csv('final_submissions/xgb_submission.csv')

list_outcome = []

for x in submission['outcome']:
    if x > .85:
        list.append(list_outcome, .95)
    elif x < .15:
        list.append(list_outcome, .05)
    else:
        list.append(list_outcome, x)

final = pd.DataFrame()
final['activity_id'] = submission['activity_id']
final['outcome'] = list_outcome
final.to_csv('final_submissions/preprocessed.csv', index=False)