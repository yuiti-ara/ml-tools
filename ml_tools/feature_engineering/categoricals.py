# import pandas as pd
#
# from sklearn.pipeline import TransformerMixin
# from sklearn_pandas import DataFrameMapper
# from sklearn_pandas.categorical_imputer import CategoricalImputer
#
#
# class FreqEncoder(TransformerMixin):
#     def __init__(self):
#         self.mapper = {}
#
#     def fit(self, x):
#         s = pd.Series(x)
#         self.mapper = s.value_counts(dropna=False, normalize=True).to_dict()
#         return self
#
#     def transform(self, x):
#         s = pd.Series(x)
#         return s.apply(lambda x: self.mapper.get(x, 0))
#
#
# def pipe_cat(cols):
#     pipe = DataFrameMapper([
#         (col, [
#             CategoricalImputer(strategy='fixed_value', replacement='_'),
#             FreqEncoder()
#         ]) for col in cols
#     ], df_out=True)
#     return pipe
