import numpy as np
import pandas as pd

from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import TransformerMixin


class SeriesLambda(TransformerMixin):
    def __init__(self, fn):
        self.fn = fn

    def fit(self, *args, **kwargs):
        return self

    def fit_transform(self, X, *args, **kwargs):
        return self.transform(X)

    def transform(self, series, *args, **kwargs):
        return self.fn(series)


def cycle_sin(values, value_max):
    return np.sin(2*np.pi*values/value_max)


def cycle_cos(values, value_max):
    return np.sin(2*np.pi*values/value_max)


def apply_dt_replace(series, year=None, month=None, day=None, hour=None, minute=None, second=None, microsecond=None):
    series_new = pd.to_datetime({
        'year': series.dt.year if year is None else year,
        'month': series.dt.month if month is None else month,
        'day': series.dt.day if day is None else day,
        'hour': series.dt.hour if hour is None else hour,
        'minute': series.dt.minute if minute is None else minute,
        'second': series.dt.second if second is None else second,
        'microsecond': series.dt.day if microsecond is None else microsecond,
    })
    return series_new


def secday(series):
    delta = series - apply_dt_replace(series, hour=0, minute=0, second=0, microsecond=0)
    return delta.dt.seconds


def weekmonth(series):
    first_day = apply_dt_replace(series, day=1)
    daymonth = series.dt.day
    adjusted_dom = daymonth + first_day.dt
    return int(np.ceil(adjusted_dom/7.0))


def tuples_cycle(cols, name, fn, value_max):
    tuples = [
        *[
            (col, [
                SeriesLambda(fn),
                SeriesLambda(lambda x: cycle_sin(x, value_max))
            ], {'input_df': True, 'alias': f'{col}_{name}_sin'})
            for col in cols
        ],
        *[
            (col, [
                SeriesLambda(fn),
                SeriesLambda(lambda x: cycle_cos(x, value_max))
            ], {'input_df': True, 'alias': f'{col}_{name}_cos'})
            for col in cols
        ],
    ]
    return tuples


def tuples_props(cols):
    pairs = [
        ('year', lambda x: x.dt.year),
        ('is_month_end', lambda x: x.dt.is_month_end),
        ('is_month_start', lambda x: x.dt.is_month_start),
        ('is_quarter_end', lambda x: x.dt.is_quarter_end),
        ('is_quarter_start', lambda x: x.dt.is_quarter_start),
        ('is_year_end', lambda x: x.dt.is_year_end),
        ('is_year_start', lambda x: x.dt.is_year_start),
    ]
    tuples = [
        (col, SeriesLambda(fn), {'input_df': True, 'alias': f'{col}_{prop}'})
        for col in cols
        for (prop, fn) in pairs
    ]
    return tuples


def pipe_time(cols):
    pipe = DataFrameMapper([
        *tuples_cycle(cols, 'secday', secday, 24*60*60),
        *tuples_cycle(cols, 'dayweek', lambda x: x.dt.dayofweek, 7),
        *tuples_cycle(cols, 'monthyear', lambda x: x.dt.month, 12),
        *tuples_props(cols),
    ], df_out=True)
    return pipe


if __name__ == '__main__':
    import datetime as dt
    df = pd.DataFrame({'date': [dt.datetime(2017, 1, 1, hour=6), dt.datetime(2017, 1, 1)]})
    pipe = pipe_time(['date'])
    df = pipe.fit_transform(df)
    print(df.to_string())
