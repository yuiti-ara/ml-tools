import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn_pandas import DataFrameMapper


def pipe_datena(df, cols):
    pipe = DataFrameMapper([
        *[
            (col, FunctionTransformer(pd.isnull, validate=False), {'alias': f'{col}_na'})
            for col in cols if df[col].isnull().sum() > 0
        ]
    ], df_out=True, input_df=True)
    return pipe


if __name__ == '__main__':
    import datetime as dt
    df = pd.DataFrame({'date1': [None, dt.datetime(2017, 1, 1), dt.datetime(2017, 1, 1)]})
    pipe = pipe_datena(df, ['date1'])
    print(pipe.fit_transform(df).to_string())
