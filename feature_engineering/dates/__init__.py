import pandas as pd
from sklearn.pipeline import TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn_pandas import DataFrameMapper

from .cycles import pipe_datecycles
from .parts import pipe_dateparts


class DateImputer(TransformerMixin):
    def __init__(self):
        self.mode = None

    def fit(self, series, _=None):
        self.mode, *_ = series.dropna().mode()

    def transform(self, series):
        series[series.isnull()] = self.mode
        return series.astype(int)


def pipe_date(df, cols):
    pipe = DataFrameMapper([
        *[
            (col, FunctionTransformer(pd.isnull, validate=False), {'alias': f'{col}_na'})
            for col in cols if df[col].isnull().sum() > 0
        ],
        *[
            (col, DateImputer())
            for col in cols if df[col].isnull().sum() > 0
        ]
    ], df_out=True, input_df=True)
    return pipe
