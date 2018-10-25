import pandas as pd

from sklearn.pipeline import TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn_pandas import DataFrameMapper


class FreqEncoder(TransformerMixin):
    def __init__(self):
        self.mapper = {}

    def fit(self, x):
        s = pd.Series(x)
        self.mapper = s.value_counts(normalize=True).to_dict()
        return self

    def transform(self, x):
        s = pd.Series(x)
        return s.apply(lambda x: self.mapper.get(x, 0))


def to_str(x):
    idx = pd.isnull(x)
    x = x.astype(str)
    x[idx] = '_'
    return x


def pipe_cat(cols):
    pipe = DataFrameMapper([
        (col, [
            FunctionTransformer(to_str, validate=False),
            FreqEncoder(),
        ]) for col in cols
    ], df_out=True)
    return pipe
