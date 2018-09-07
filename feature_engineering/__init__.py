from sklearn_pandas import DataFrameMapper

from .pandas_feature_union import PandasFeatureUnion
from .categoricals import pipe_cat
from .numericals import pipe_num
from .dates import pipe_datecycles, pipe_dateparts, pipe_date
from .dummies import pipe_dummy


def pipe_raw(cols):
    return DataFrameMapper([(col, None) for col in cols], df_out=True)
