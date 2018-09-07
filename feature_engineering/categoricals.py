from sklearn_pandas import DataFrameMapper
from sklearn_pandas.categorical_imputer import CategoricalImputer
from category_encoders import OrdinalEncoder


def pipe_cat(cols):
    pipe = DataFrameMapper([
        (col, [
            CategoricalImputer(strategy='fixed_value', replacement='_'),
            OrdinalEncoder()
        ]) for col in cols
    ], df_out=True)
    return pipe
