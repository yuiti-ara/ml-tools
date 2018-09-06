from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper

from .pandas_feature_union import PandasFeatureUnion
from .pipe_pre import pipe_pre
from .pipe_time import pipe_time
from .pipe_dummy import pipe_dummy


def pipe_all(label='all'):
    return Pipeline([(label, None)])


def pipe_raw(cols):
    return DataFrameMapper([(col, None) for col in cols], df_out=True)
