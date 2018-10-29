from eli5.sklearn import PermutationImportance
from sklearn.pipeline import TransformerMixin


class PermImpElimination(TransformerMixin):
    def __init__(self, model, cutoff=0):
        self.model = model
        self.cutoff = cutoff
        self.pimp = PermutationImportance(self.model, cv=4, random_state=42)
        self.cols_selected = None

    def fit(self, df, y):
        cols_good = list(df.columns)

        while cols_good:
            self.pimp.fit(df[cols_good], y)
            imps = self.pimp.feature_importances_
            idx_bad = imps.argmin()
            value_bad = imps[idx_bad]
            if value_bad > self.cutoff:
                break

            col_dropped = cols_good.pop(idx_bad)
            print(f'dropped: {col_dropped} {value_bad}')

        self.cols_selected = cols_good
        return self

    def transform(self, df):
        return df[self.cols_selected]
