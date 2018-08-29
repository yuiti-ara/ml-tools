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

    def transform(self, X, *args, **kwargs):
        df = pd.DataFrame(X)
        for col in df:
            df[col] = self.fn(df[col])
        return df


def datetime_pipe(cols):
    fns = [
        lambda x: x.dt.year,
        lambda x: x.dt.month,
        lambda x: x.dt.day,
        lambda x: x.dt.dayofweek,
        lambda x: x.dt.dayofyear,
        lambda x: x.dt.is_month_end,
        lambda x: x.dt.is_month_start,
        lambda x: x.dt.is_quarter_end,
        lambda x: x.dt.is_quarter_start,
        lambda x: x.dt.is_year_end,
        lambda x: x.dt.is_year_start,
    ]
    # TODO
    pipe = DataFrameMapper([
        (cols, LambdaTransformer(fn)) for fn in fns
    ], df_out=True)
    return pipe


if __name__ == '__main__':
    import datetime as dt
    df = pd.DataFrame({'date': [dt.datetime(2017, 1, 1), dt.datetime(2016, 1, 1)]})

    pipe = datetime_pipe(['date'])

    print(pipe.fit_transform(df))
