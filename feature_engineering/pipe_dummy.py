from sklearn_pandas import DataFrameMapper
from category_encoders import OneHotEncoder


def pipe_dummy(cols):
    pipe = DataFrameMapper([
        (cols, OneHotEncoder(use_cat_names=True, impute_missing=False, drop_invariant=True))
    ], df_out=True)
    return pipe
