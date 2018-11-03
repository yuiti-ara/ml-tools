from IPython.display import display

import pandas as pd
import scikitplot as skplt
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, roc_auc_score, log_loss, f1_score, make_scorer
from rfpimp import importances
from joblib import Parallel, delayed


def evaluate(model, X_tr, y_tr, X_vl, y_vl, metric=True, report=False, cm=False, roc=False, imp=False):
    model.fit(X_tr, y_tr)
    y_tr_pred = model.predict(X_tr)
    y_tr_proba = model.predict_proba(X_tr)
    y_tr_score = y_tr_proba[:, 1]
    
    y_vl_pred = model.predict(X_vl)
    y_vl_proba = model.predict_proba(X_vl)
    y_vl_score = y_vl_proba[:, 1]
    
    if metric:
        index = ['roc_auc', 'log_loss', 'f1-score']
        metrics = {
            'tr': [roc_auc_score(y_tr, y_tr_score), log_loss(y_tr, y_tr_score), f1_score(y_tr, y_tr_pred)],
            'vl': [roc_auc_score(y_vl, y_vl_score), log_loss(y_vl, y_vl_score), f1_score(y_vl, y_vl_pred)],
        }
        display(pd.DataFrame(metrics, index=index))
    
    if report:
        report = classification_report(y_vl, y_vl_pred, output_dict=True)
        display(pd.DataFrame(report)[['0', '1']])
    
    if cm:
        skplt.metrics.plot_confusion_matrix(y_vl, y_vl_pred)
        
    if roc:
        skplt.metrics.plot_roc(y_vl, y_vl_proba, classes_to_plot=['1'])
        
    if imp:
        imps = bootstrapped_imps(model, X_tr, y_tr, X_vl, y_vl)
        display(pd.DataFrame(imps.sort_values(ascending=False)))
        
        perm = PermutationImportance(model, scoring=make_scorer(roc_auc_score), random_state=42)
        perm.fit(X_vl, y_vl)
        display(eli5.show_weights(perm, feature_names=list(X_vl.columns)))


def evaluate_cv(model, X, y):
    scores = cross_validate(model, X, y, cv=4, scoring=['roc_auc', 'neg_log_loss', 'f1'], n_jobs=-1, return_train_score=True)
    df_scores = pd.DataFrame(scores)
    df_cv = pd.DataFrame()
    df_cv['cv_avg'] = df_scores.mean()
    df_cv['cv_std'] = df_scores.std()
    display(df_cv.iloc[::-1])


def fn_imp(model, X_vl, y_vl):
    imp = importances(model, X_vl, y_vl, metric=make_scorer(roc_auc_score), sort=False)
    return imp['Importance']


def bootstrapped_imps(model, X_tr, y_tr, X_vl, y_vl, n_iter=5):
    model.fit(X_tr, y_tr)
    imps = Parallel(n_jobs=-1)(delayed(fn_imp)(model, X_vl, y_vl) for _ in range(n_iter))
    df_imps = pd.DataFrame(imps).transpose()
    series = df_imps.sum(axis='columns')
    return series/series.sum()
