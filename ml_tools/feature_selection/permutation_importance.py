import pandas as pd
from sklearn.pipeline import TransformerMixin
from sklearn.metrics import make_scorer, roc_auc_score 
from rfpimp import importances
from eli5.sklearn import PermutationImportance
from joblib import Parallel, delayed


class PermImpEliminationEli5(TransformerMixin):
    def __init__(self, model, X_tr, y_tr, cutoff=0):
        self.model = model
        self.X_tr = X_tr
        self.y_tr = y_tr
        self.scorer = make_scorer(roc_auc_score)
        self.cutoff = cutoff
        self.cols_selected = None

    def fit(self, X_vl, y_vl):
        cols_good = list(self.X_tr.columns)

        while cols_good:
            self.model.fit(self.X_tr[cols_good], self.y_tr)
            perm = PermutationImportance(self.model, scoring=self.scorer, random_state=42)
            perm.fit(X_vl[cols_good], y_vl)
            imps = perm.feature_importances_
            
            idx_bad = imps.argmin()
            value_bad = imps[idx_bad]
            if value_bad > self.cutoff:
                break

            col_dropped = cols_good.pop(idx_bad)
            print(f'dropped: {col_dropped} {value_bad}')

        self.cols_selected = cols_good
        return self

    def transform(self, X_vl):
        return X_vl[self.cols_selected]


class PermImpElimination(TransformerMixin):
    def __init__(self, model, X_tr, y_tr, cutoff=0):
        self.model = model
        self.X_tr = X_tr
        self.y_tr = y_tr
        self.scorer = make_scorer(roc_auc_score)
        self.cutoff = cutoff
        self.cols_selected = None

    def fit(self, X_vl, y_vl):
        cols_good = list(self.X_tr.columns)

        while cols_good:
            series_imp = bootstrapped_imps(self.model, self.X_tr[cols_good], self.y_tr, X_vl[cols_good], y_vl)
            col_bad = series_imp.idxmin()
            value_bad = series_imp[col_bad]
            if value_bad > self.cutoff:
                break

            cols_good.remove(col_bad)
            print(f'dropped: {col_bad} {value_bad}')

        self.cols_selected = cols_good
        return self

    def transform(self, X_vl):
        return X_vl[self.cols_selected]


def fn_imp(model, X_vl, y_vl):
    imp = importances(model, X_vl, y_vl, metric=make_scorer(roc_auc_score), sort=False)
    return imp['Importance']


def bootstrapped_imps(model, X_tr, y_tr, X_vl, y_vl, n_iter=5):
    model.fit(X_tr, y_tr)
    imps = Parallel(n_jobs=-1)(delayed(fn_imp)(model, X_vl, y_vl) for _ in range(n_iter))
    df_imps = pd.DataFrame(imps).transpose()
    series = df_imps.sum(axis='columns')
    return series/series.sum()
