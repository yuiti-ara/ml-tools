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


def secday(series):
    delta = series - dt_replace(series, hour=0, minute=0, second=0, microsecond=0)
    return delta.dt.seconds


def weekmonth(series):
    first_day = dt_replace(series, day=1)
    daymonth = series.dt.day
    adjusted_dom = daymonth + first_day.dt
    return int(np.ceil(adjusted_dom/7.0))


class DateCycle(TransformerMixin):
    def __init__(self, datepart_fn, value_max):
        self.prefix = None
        self.datepart_name = datepart_fn.__name__
        self.datepart_fn = datepart_fn
        self.value_max = value_max

    def fit(self, x, _=None):
        self.prefix = getattr(x, 'name', None)
        return self

    def transform(self, x):
        series = self.datepart_fn(pd.Series(x))
        s_sin = np.sin(2*np.pi*series/self.value_max)
        s_cos = np.cos(2*np.pi*series/self.value_max)
        df = pd.DataFrame({
            f'{self.prefix}_{self.datepart_name}_sin': s_sin,
            f'{self.prefix}_{self.datepart_name}_cos': s_cos
        })
        return df


def pipe_datecycles(cols):
    pipe = DataFrameMapper([
        *[(col, [DateCycle(secday, 24*60*60)], {'alias': f'{col}_secday_cycle'}) for col in cols],
        *[(col, [DateCycle(lambda x: x.dt.dayofweek, 7)], {'alias': f'{col}_dayofweek_cycle'}) for col in cols],
        *[(col, [DateCycle(lambda x: x.dt.month, 12)], {'alias': f'{col}_month_cycle'}) for col in cols],
    ], df_out=True)
    return pipe


if __name__ == '__main__':
    import datetime as dt
    df = pd.DataFrame({'date1': [None, dt.datetime(2017, 1, 1), dt.datetime(2017, 1, 1)]})
    pipe = pipe_datecycles(['date1'])
    print(pipe.fit_transform(df).to_string())
