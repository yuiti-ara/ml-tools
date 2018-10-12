from itertools import permutations

import pandas as pd
from sklearn.pipeline import TransformerMixin
from rfpimp import importances


class FilterCorr(TransformerMixin):
    def __init__(self):
        self.cols_selected = []

    def fit(self, df):
        df_corr = df.corr(method='spearman').abs()

        groups = []
        for idx, (col, row) in enumerate(df_corr.iterrows()):
            row = row[idx + 1:]
            row = row[row > .5]
            if not row.empty:
                group = {col, *row.index}
                groups.append(group)

        to_drop = set()
        for group in groups:
            df_combs = pd.DataFrame(list(permutations(group, 2)))

            def fn(x):
                corr = df_corr.loc[x[0], x[1]]
                if corr > .5:
                    return corr

            df_combs['corr'] = df_combs.apply(fn, axis='columns')

            df_counts = df_combs.groupby(0)['corr'].agg(['count', 'sum'])
            df_counts = df_counts.sort_values(['count', 'sum'], ascending=False)

            row = df_counts.iloc[0, :]
            if row['count'] == len(df_counts) - 1:
                to_drop |= set(df_counts.iloc[1:, 0].index)

        self.cols_selected = list(set(df.columns) - to_drop)
        return self

    def transform(self, df):
        return df[self.cols_selected]


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
