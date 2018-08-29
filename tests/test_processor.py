import datetime as dt

import numpy as np
import pandas as pd

from run import get_preprocessor


class TestPreProcessor:
    def test_pre_processor(self):
        proc = get_preprocessor(cats=['cat1', 'cat2'], nums=['num1', 'num2'], dates=['date1', 'date2'])

        data_in = {
            'cat1': ['a', 'b', 'c', 'd', 'e'],
            'cat2': ['a', 'b', None, None, 'e'],
            'num1': [1, 2.5, 3, 4, 5.2],
            'num2':  [1, 2.5, None, None, 5.2],
            'date1': [dt.datetime(2017, 1, idx) for idx in range(1, 6)],
            'date2': [None, *[dt.datetime(2017, 1, idx) for idx in range(1, 4)], None],
        }
        df_in = pd.DataFrame(data_in)

        df_new = df_in.copy()
        df_new['cat1'] = [0, 1, 2, 3, 4]
        df_new['cat2'] = [1, 2, 0, 0, 3]
        num2 = df_new['num2']
        df_new['num2'] = (num2 - num2.median())/num2.quantile(.75)

        print(df_new, num2.quantile(.75))

