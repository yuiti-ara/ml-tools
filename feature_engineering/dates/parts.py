import pandas as pd

from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import TransformerMixin


class DatePart(TransformerMixin):
    def __init__(self, fn):
        self.fn = fn

    def fit(self, *args, **kwargs):
        return self

    def transform(self, x):
        return self.fn(pd.Series(x))


def pipe_dateparts(cols):
    props = [
        'year',
        'is_month_end',
        'is_month_start',
        'is_quarter_end',
        'is_quarter_start',
        'is_year_end',
        'is_year_start',
    ]
    pipe = DataFrameMapper([
        (col, DatePart(lambda x: getattr(x, prop)), {'alias': f'{col}_{prop}'})
        for col in cols
        for prop in props
    ], df_out=True)
    return pipe
