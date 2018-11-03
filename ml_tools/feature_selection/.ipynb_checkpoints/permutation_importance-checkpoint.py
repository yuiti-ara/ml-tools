from eli5.sklearn import PermutationImportance
from sklearn.pipeline import TransformerMixin
from sklearn.metrics import make_scorer, roc_auc_score 
from rfpimp import importances


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


def bootstrapped_imps(model, X_tr, y_tr, X_vl, y_vl, n_iter=5):
    model.fit(X_tr, y_tr)
    imps = importances(model, X_vl, y_vl, sort=False)
    return imps
