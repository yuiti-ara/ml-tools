from sklearn.pipeline import TransformerMixin
from rfpimp import importances


class FilterCorr(TransformerMixin):
    def __init__(self):
        self.cols = []

    def fit(self, df):
        df_corr = df.corr(method='spearman')

        cols_bad = set()
        for idx, (col, row) in enumerate(df_corr.iterrows()):
            series = row[idx + 1:].abs() > .5
            if series.sum() > 0:
                cols_bad.add(col)
        self.cols = list(set(df.columns) - cols_bad)

        return self

    def transform(self, df):
        return df[self.cols]


def rfe(model, df_tr, y_tr, df_vl, y_vl):
    while True:
        model.fit(df_tr, y_tr)
        imps = importances(model, df_vl, y_vl)
        row = imps.iloc[-1]
        col, delta = row.name, row.values[0]
        if delta > 0:
            return model, imps
        df_tr = df_tr.drop(col, axis='columns')
        df_vl = df_vl.drop(col, axis='columns')
        print(f'{df_tr.shape} dropped: {col} {delta}')
