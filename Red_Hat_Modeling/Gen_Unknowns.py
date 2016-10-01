__author__ = "Nick Sarris"

import pandas as pd
import numpy as np

data_leak = ('data_leak/data_leak.csv')
leak = pd.read_csv(data_leak)

unknowns = leak[leak['outcome'].apply(np.isnan)]
knowns = leak[np.isfinite(leak['outcome'])]

knowns.to_csv('data_leak/knowns.csv', index=False)
unknowns.to_csv('data_leak/unknowns.csv', index=False)

