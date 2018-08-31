import numpy as np
import pandas as pd
from sklearn_pandas.categorical_imputer import CategoricalImputer
from sklearn.preprocessing import Imputer, FunctionTransformer, RobustScaler, LabelEncoder


class ModifiedLabelEncoder(LabelEncoder):
    def fit(self, y, *args, **kwargs):
        return super().fit(y)

    def fit_transform(self, y, *args, **kwargs):
        return super().fit_transform(y).reshape(-1, 1)

    def transform(self, y, *args, **kwargs):
        return super().transform(y).reshape(-1, 1)


def tuples_pre(df, cats=None, nums=None, dates=None):
    cats = cats or []
    nums = nums or []
    dates = dates or []
    pipe = [
        *[
            (col, [CategoricalImputer(strategy='fixed_value', replacement='_'), ModifiedLabelEncoder()])
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
        *[
            (col, [FunctionTransformer(np.int64, validate=False)])
            for col in dates
        ],

    ]
    return pipe


if __name__ == '__main__':
    import datetime as dt
    from sklearn_pandas import DataFrameMapper
    data_in = {
        'cat1': ['a', 'b', 'c', 'd', 'e'],
        'cat2': ['a', 'b', None, None, 'e'],
        'num1': [1, 2.5, 3, 4, 5.2],
        'num2': [1, 2.5, None, None, 5.2],
        'date1': [dt.datetime(2017, 1, idx) for idx in range(1, 6)],
        'date2': [None, *[dt.datetime(2017, 1, idx) for idx in range(1, 4)], None],
    }
    df_in = pd.DataFrame(data_in)

    pipe = DataFrameMapper(
        tuples_pre(df_in, cats=['cat1', 'cat2'], nums=['num1', 'num2'], dates=['date1', 'date2']),
    df_out=True)
    print(pipe.fit_transform(df_in).to_string())
