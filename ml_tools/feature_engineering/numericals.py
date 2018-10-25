from sklearn_pandas import DataFrameMapper
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def reshape(x):
    return x.reshape(-1, 1)


def pipe_num(df, cols):
    pipe = DataFrameMapper([
        *[
            (col, [
                FunctionTransformer(reshape, validate=False),
                MissingIndicator(),
            ], {'alias': f'{col}_na'}) for col in cols if df[col].isnull().sum() > 0
        ],
        *[
            (col, [
                FunctionTransformer(reshape, validate=False),
                SimpleImputer(strategy='median'),
                StandardScaler()
            ]) for col in cols
        ]
    ], df_out=True)
    return pipe
