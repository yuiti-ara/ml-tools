import pandas as pd
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import Imputer, FunctionTransformer, RobustScaler


def pipe_num(df, cols):
    pipe = DataFrameMapper([
        *[
            (col, FunctionTransformer(pd.isnull, validate=False), {'alias': f'{col}_na'})
            for col in cols if df[col].isnull().sum() > 0
        ],
        *[
            (col, [
                FunctionTransformer(lambda x: x.reshape(-1, 1), validate=False),
                Imputer(strategy='median'),
                RobustScaler()
            ]) for col in cols
        ]
    ], df_out=True)
    return pipe
