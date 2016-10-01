__author__ = "Nick Sarris"

import pandas as pd

knowns = pd.read_csv('data_leak/knowns.csv')
knowns.columns = (['activity_id', 'outcome'])
activity_id = knowns.activity_id.tolist()
model_output = pd.read_csv('model_output/xgboost_submission.csv')
model_output = model_output[~model_output['activity_id'].isin(activity_id)]
final_submission = pd.concat([knowns,  model_output], axis=0, ignore_index=True)
final_submission.drop_duplicates(take_last=True)
print (final_submission)

final_submission.to_csv('final_submissions/xgb_submission.csv', index=False)