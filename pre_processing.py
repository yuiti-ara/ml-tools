import numpy as np
import pandas as pd
from sklearn_pandas import DataFrameMapper
from sklearn_pandas.categorical_imputer import CategoricalImputer
from sklearn.pipeline import TransformerMixin
from sklearn.preprocessing import Imputer, FunctionTransformer, RobustScaler, LabelEncoder


class ModifiedLabelEncoder(LabelEncoder):
    def fit(self, y, *args, **kwargs):
        return super().fit(y)

    def fit_transform(self, y, *args, **kwargs):
        return super().fit_transform(y).reshape(-1, 1)

    def transform(self, y, *args, **kwargs):
        return super().transform(y).reshape(-1, 1)


def pre_processor(df, cats=None, nums=None, dates=None):
    cats = cats or []
    nums = nums or []
    dates = dates or []
    pipe = DataFrameMapper([
        *[
            (col, [CategoricalImputer(strategy='fixed_value', replacement='_'), ModifiedLabelEncoder()])
            for col in cats
        ],
        *[
            (col, [FunctionTransformer(pd.isnull, validate=False)], {'alias': f'{col}_na'})
            for col in nums+dates if df[col].isnull().sum() > 0
        ],
        *[
            (col, [
                FunctionTransformer(lambda x: x.reshape(-1, 1), validate=False),
                Imputer(strategy='median'),
                RobustScaler()
            ])
            for col in nums
        ],
        *[
            (col, [FunctionTransformer(np.int64, validate=False)])
            for col in dates
        ],

    ], df_out=True)
    return pipe


class LambdaTransformer(TransformerMixin):
    def __init__(self, fn):
        self.fn = fn

    def fit(self, *args, **kwargs):
        return self

    def fit_transform(self, X, *args, **kwargs):
        return self.transform(X)

    def transform(self, series, *args, **kwargs):
        return self.fn(series)


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


def cycle_sin(series):
    delta = series - apply_dt_replace(series, hour=0, minute=0, second=0, microsecond=0)
    secs_of_day = delta.dt.seconds
    value_max = 24*60*60
    return np.sin(2*np.pi*secs_of_day/value_max)


def datetime_pipe(cols):
    props = [
        'year',
        'month',
        'day',
        'hour',
        'dayofweek',
        'dayofyear',
        'is_month_end',
        'is_month_start',
        'is_quarter_end',
        'is_quarter_start',
        'is_year_end',
        'is_year_start',
    ]
    fns = [(prop, lambda x: getattr(x.dt, prop)) for prop in props]
    pipe = DataFrameMapper([
        (col, LambdaTransformer(fn), {'input_df': True, 'alias': f'{col}_{fn_name}'})
        for col in cols
        for (fn_name, fn) in fns
    ], df_out=True)
    return pipe


if __name__ == '__main__':
    import datetime as dt
    df = pd.DataFrame({'date': [dt.datetime(2017, 1, 1, 1), dt.datetime(2017, 1, 1, 2)]})
    s = cycle_sin(df['date'])
    print(s)
