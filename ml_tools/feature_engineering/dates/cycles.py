import numpy as np
import pandas as pd

from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import TransformerMixin


def dt_replace(series, year=None, month=None, day=None, hour=None, minute=None, second=None, microsecond=None):
    series_new = pd.to_datetime({
        'year': year or series.dt.year,
        'month': month or series.dt.month,
        'day': day or series.dt.day,
        'hour': hour or series.dt.hour,
        'minute': minute or series.dt.minute,
        'second': second or series.dt.second,
        'microsecond': microsecond or series.dt.microsecond,
    })
    return series_new


def sec_of_day(series):
    delta = series - dt_replace(series, hour=0, minute=0, second=0, microsecond=0)
    return delta.dt.seconds


def week_of_month(series):
    first_day = dt_replace(series, day=1)
    dom = series.dt.day
    adjusted_dom = dom + first_day.dt.weekday
    return np.int64(np.ceil(adjusted_dom/7.0))


class DateCycle(TransformerMixin):
    def __init__(self, name):
        self.name = name
        self.prefix = None

    def fit(self, x):
        self.prefix = getattr(x, 'name', None)
        return self

    @staticmethod
    def _fn(name, x):
        x = pd.Series(x)
        periods = {
            'sec_day': (sec_of_day, 24*60*60),
            'day_week': (lambda x: x.dt.dayofweek, 7),
            'week_month': (week_of_month, x.dt.daysinmonth),
            'month_year': (lambda x: x.dt.month, 12),
        }
        fn, value_max = periods[name]
        return fn(x), value_max

    def transform(self, x):
        series, value_max = self._fn(self.name, x)
        s_sin = np.sin(2*np.pi*series/value_max)
        s_cos = np.cos(2*np.pi*series/value_max)
        return pd.DataFrame({'sin': s_sin, 'cos': s_cos})


def pipe_datecycles(cols):
    pipe = DataFrameMapper([
        *[(col, [DateCycle('sec_day')], {'alias': f'{col}_cycle_sec_day'}) for col in cols],
        *[(col, [DateCycle('day_week')], {'alias': f'{col}_cycle_day_week'}) for col in cols],
        *[(col, [DateCycle('week_month')], {'alias': f'{col}_cycle_week_month'}) for col in cols],
        *[(col, [DateCycle('month_year')], {'alias': f'{col}_cycle_month_year'}) for col in cols],
    ], df_out=True)
    return pipe


if __name__ == '__main__':
    import datetime as dt
    df = pd.DataFrame({'date1': [None, dt.datetime(2018, 10, 26), dt.datetime(2017, 1, 1)]})
    pipe = pipe_datecycles(['date1'])
    print(pipe.fit_transform(df).to_string())
