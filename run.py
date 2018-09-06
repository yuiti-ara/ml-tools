import datetime as dt
import pandas as pd

from feature_engineering import Pipeline, PandasFeatureUnion, pipe_all, pipe_pre, pipe_time, pipe_dummy, pipe_raw


data_in = {
    'cat1': ['a', 'b', 'c', 'd', 'e'],
    'cat2': ['a', 'b', None, None, 'e'],
    'num1': [1, 2.5, 3, 4, 5.2],
    'num2': [1, 2.5, None, None, 5.2],
    'date1': [dt.datetime(2017, 1, idx) for idx in range(1, 6)],
    'date2': [None, *[dt.datetime(2017, 1, idx) for idx in range(1, 4)], None],
}
df_in = pd.DataFrame(data_in)


cats = ['cat1', 'cat2']
nums = ['num1', 'num2']
dates = ['date1', 'date2']

from sklearn.pipeline import make_pipeline

pipe = PandasFeatureUnion([
    ('filled', pipe_pre(df_in, cats=cats, nums=nums, dates=dates)),
    ('dummy', pipe_dummy(cols=['cat1'])),
    ('datetime', pipe_time(cols=dates))
])

pipe.fit(df_in)
X = pipe.transform(df_in)
print(X.to_string())
