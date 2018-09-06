import numpy as np
import pandas as pd
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import Imputer, FunctionTransformer, RobustScaler
from sklearn_pandas.categorical_imputer import CategoricalImputer
from category_encoders import OrdinalEncoder


def pipe_pre(df, cats=None, nums=None, dates=None):
    cats = cats or []
    nums = nums or []
    dates = dates or []
    tuples = [
        *[
            (col, [CategoricalImputer(strategy='fixed_value', replacement='_'), OrdinalEncoder()])
            for col in cats
        ],
        *[
            (col, [FunctionTransformer(pd.isnull, validate=False)], {'alias': f'{col}_na'})
            for col in nums+dates if df[col].isnull().sum() > 0
        ],
        *[
            (col, [
                FunctionTransformer(lambda x: x.reshape(-1, 1), validate=False),
                Imputer(strategy='median'),
                RobustScaler()
            ])
            for col in nums
        ],
        *[(col, FunctionTransformer(np.int64, validate=False)) for col in dates],
    ]
    return DataFrameMapper(tuples, df_out=True)
