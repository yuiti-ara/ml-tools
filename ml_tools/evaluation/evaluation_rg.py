from IPython.display import display

import pandas as pd
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error, r2_score


def evaluate(model, X_tr, y_tr, X_vl, y_vl, metric=False, imp=False):
    model.fit(X_tr, y_tr)
    y_tr_pred = model.predict(X_tr)
    y_vl_pred = model.predict(X_vl)
    
    if metric:
        index = ['mae', 'r2']
        metrics = {
            'tr': [mean_absolute_error(y_tr, y_tr_pred), r2_score(y_tr, y_tr_pred)],
            'vl': [mean_absolute_error(y_vl, y_vl_pred), r2_score(y_vl, y_vl_pred)],
        }
        display(pd.DataFrame(metrics, index=index))
        
    if imp:
        pimp = PermutationImportance(model, random_state=42, n_iter=20)
        pimp.fit(X_vl, y_vl)
        display(eli5.show_weights(pimp))


def evaluate_cv(model, X, y):
    scores = cross_validate(model, X, y, cv=4, scoring=['neg_mean_absolute_error', 'r2'], n_jobs=-1, return_train_score=True)
    df_scores = pd.DataFrame(scores)
    df_cv = pd.DataFrame()
    df_cv['cv_avg'] = df_scores.mean()
    df_cv['cv_std'] = df_scores.std()
    display(df_cv.iloc[::-1])
