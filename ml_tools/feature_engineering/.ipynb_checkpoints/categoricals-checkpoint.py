import pandas as pd

from sklearn.pipeline import TransformerMixin
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn_pandas import DataFrameMapper


class FreqEncoder(TransformerMixin):
    def __init__(self, min_qty=0):
        self.mapper = {}
        self.min_qty = min_qty
        
    def fit(self, x):
        mapper = pd.Series(x).value_counts().to_dict()
        mapper = {k: v/len(x) for k, v in mapper.items() if v >= self.min_qty}
        self.mapper = mapper
        return self

    def transform(self, x):
        s = pd.Series(x)
        return s.apply(lambda x: self.mapper.get(x, 0))


def _to_str(x):
    idx = pd.isnull(x)
    x = x.astype(str)
    x[idx] = '_'
    return x


def _reshape(x):
    return x.reshape(-1, 1)


def pipe_cat(cols_big, cols_small):
    pipe = DataFrameMapper([
        *[
            (col, [
                FunctionTransformer(_to_str, validate=False),
                FreqEncoder(),
            ]) for col in cols_big
        ],
        *[
            (col, [
                FunctionTransformer(_to_str, validate=False),
                FunctionTransformer(_reshape, validate=False),
                OneHotEncoder(),
            ]) for col in cols_small
        ],
    ], df_out=True)
    return pipe
