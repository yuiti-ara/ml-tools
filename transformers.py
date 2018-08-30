from sklearn.pipeline import TransformerMixin
from sklearn.preprocessing import LabelEncoder


class ModifiedLabelEncoder(LabelEncoder):
    def fit(self, y, *args, **kwargs):
        return super().fit(y)

    def fit_transform(self, y, *args, **kwargs):
        return super().fit_transform(y).reshape(-1, 1)

    def transform(self, y, *args, **kwargs):
        return super().transform(y).reshape(-1, 1)


class SeriesLambda(TransformerMixin):
    def __init__(self, fn):
        self.fn = fn

    def fit(self, *args, **kwargs):
        return self

    def fit_transform(self, X, *args, **kwargs):
        return self.transform(X)

    def transform(self, series, *args, **kwargs):
        return self.fn(series)
