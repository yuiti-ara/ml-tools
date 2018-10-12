import datetime as dt
import pandas as pd

from feature_engineering import (
    PandasFeatureUnion,
    pipe_cat,
    pipe_num,
    pipe_datena,
    pipe_dateparts,
    pipe_datecycles,
    pipe_raw,
)

df_tr = pd.DataFrame({
    'cat1': ['a', 'b', 'c', 'd', 'e'],
    'cat2': ['a', 'b', None, None, 'e'],
    'num1': [1, 2.5, 3, 4, 5.2],
    'num2': [1, 2.5, None, None, 5.2],
    'date1': [dt.datetime(2017, 1, idx) for idx in range(1, 6)],
    'date2': [None, *[dt.datetime(2017, 1, idx) for idx in range(1, 4)], None],
})

df_ts = pd.DataFrame({
    'cat1': [None, None],
    'cat2': [None, None],
    'num1': [None, None],
    'num2': [None, None],
    'date1': [None, None],
    'date2': [None, None],
})


cats = ['cat1', 'cat2']
nums = ['num1', 'num2']
dates = ['date1', 'date2']

pipe = PandasFeatureUnion([
    ('cats', pipe_cat(cats)),
    ('nums', pipe_num(df_tr, nums)),
    ('cycles', pipe_datecycles(['date2'])),
    ('raw', pipe_raw(cats))
])

pipe.fit(df_tr)
X_tr = pipe.transform(df_tr)
print(X_tr.to_string())

# X_ts = pipe.transform(df_ts)
# print(X_ts.to_string())
