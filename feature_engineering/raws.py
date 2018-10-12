from sklearn_pandas import DataFrameMapper


def pipe_raw(cols):
    return DataFrameMapper([(col, None) for col in cols], df_out=True)
