import pandas as pd

from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import TransformerMixin


class DatePart(TransformerMixin):
    def __init__(self, datepart):
        self.datepart = datepart

    def fit(self, *args, **kwargs):
        return self

    def transform(self, x):
        return getattr(pd.Series(x).dt, self.datepart)


def pipe_dateparts(cols):
    props = [
        'year',
        'month',
        'day',
        'hour',
        'minute',
        'second',
        'microsecond',
        'nanosecond',
        'week',
        'weekofyear',
        'dayofweek',
        'weekday',
        'dayofyear',
        'quarter',
        'is_month_start',
        'is_month_end',
        'is_quarter_start',
        'is_quarter_end',
        'is_year_start',
        'is_year_end',
        'is_leap_year',
        'daysinmonth',
        'days_in_month',
    ]
    pipe = DataFrameMapper([
        (col, DatePart(prop), {'alias': f'{col}_{prop}'})
        for col in cols
        for prop in props
    ], df_out=True)
    return pipe


if __name__ == '__main__':
    import datetime as dt
    df = pd.DataFrame({'date1': [None, dt.datetime(2017, 1, 1), dt.datetime(2017, 1, 1)]})
    pipe = pipe_dateparts(['date1'])
    print(pipe.fit_transform(df).to_string())
