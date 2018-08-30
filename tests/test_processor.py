import datetime as dt

import pandas as pd

from pipe_pre import pipe_pre


class TestPreProcessor:
    def test_pre_processor(self):
        data_in = {
            'cat1': ['a', 'b', 'c', 'd', 'e'],
            'cat2': ['a', 'b', None, None, 'e'],
            'num1': [1, 2.5, 3, 4, 5.2],
            'num2': [1, 2.5, None, None, 5.2],
            'date1': [dt.datetime(2017, 1, idx) for idx in range(1, 6)],
            'date2': [None, *[dt.datetime(2017, 1, idx) for idx in range(1, 4)], None],
        }
        df_in = pd.DataFrame(data_in)

        pipe = pipe_pre(
            df_in,
            cats=['cat1', 'cat2'],
            nums=['num1', 'num2'],
            dates=['date1', 'date2']
        )
        df_out = pipe.fit_transform(df_in)

        cols_expected = {
            'cat1',
            'cat2',
            'num1',
            'num2',
            'date1',
            'date2',
            'num2_na',
            'date2_na',
        }
        assert set(df_out) == cols_expected
