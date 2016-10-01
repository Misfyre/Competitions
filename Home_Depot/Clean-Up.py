import numpy as np
import pandas as pd
import os

submission = pd.read_csv(os.path.expanduser('~/Desktop/Home Depot/Output/Submissions/submission.csv'))
submission_1 = pd.read_csv(os.path.expanduser('~/Desktop/Home Depot/Output/Submissions/submission_1.csv'))
submission_2 = pd.read_csv(os.path.expanduser('~/Desktop/Home Depot/Output/Submissions/submission_2.csv'))

def fix_outliers(input):

    if input > 3:
        input = 3.0
    elif input < 1:
        input = 1.0
    return input

submission['relevance'] = submission['relevance'].map(lambda x:fix_outliers(x))
submission_1['relevance'] = submission_1['relevance'].map(lambda x:fix_outliers(x))
submission_2['relevance'] = submission_2['relevance'].map(lambda x:fix_outliers(x))

submission.to_csv(os.path.expanduser('~/Desktop/Home Depot/Output/Final/final_submission.csv'), index=False)
submission_1.to_csv(os.path.expanduser('~/Desktop/Home Depot/Output/Final/final_submission_1.csv'), index=False)
submission_2.to_csv(os.path.expanduser('~/Desktop/Home Depot/Output/Final/final_submission_2.csv'), index=False)