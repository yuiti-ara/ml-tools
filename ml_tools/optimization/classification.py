import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, cross_val_score
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from parfit import bestFit


def optimize_xg(model_xg, X_tr, y_tr, X_vl=None, y_vl=None):
    space = [
        Integer(50, 500, name='n_estimators'),
        Integer(1, 5, name='max_depth'),
        Integer(1, 10, name='min_child_weight'),
        Real(10 ** -5, 1, "log-uniform", name='learning_rate'),
        Integer(1, 10, name='max_delta_step'),

        Real(.1, 1, name='subsample'),
        Real(.1, 1, name='colsample_bytree'),

        Real(10 ** -5, 1, "log-uniform", name='reg_alpha'),
        Real(10 ** -5, 1, "log-uniform", name='reg_lambda'),
        Real(10 ** -5, 10, "log-uniform", name='gamma'),
    ]

    @use_named_args(space)
    def objective(**params):
        model_xg.set_params(**params)

        if X_vl is None or y_vl is None:
            scores = cross_val_score(model_xg, X_tr, y_tr, cv=4, n_jobs=-1, scoring="roc_auc")
            score = np.mean(scores)
        else:
            model_xg.fit(X_tr, y_tr)
            y_vl_score = model_xg.predict_proba(X_vl)[:, 1]
            score = roc_auc_score(y_vl, y_vl_score)

        return -score

    resp = gp_minimize(
        objective,
        space,
        n_calls=100,
        random_state=42,
        verbose=True,
        n_jobs=-1,
    )

    return resp


def optimize_rf(model_rf, X_tr, y_tr, X_vl=None, y_vl=None):
    if X_vl is None or y_vl is None:
        kwargs = {
            'X_train': X_tr,
            'y_train': y_tr,
            'nfolds': 4,
        }
    else:
        kwargs = {
            'X_train': X_tr,
            'y_train': y_tr,
            'X_val': X_vl,
            'y_val': y_vl,
        }

    best_model, best_score, all_models, all_scores = bestFit(
        model_rf,
        ParameterGrid({
            'min_samples_leaf': [10, 25, 50, 75, 100],
            'max_features': [1, .75, .5, 'sqrt', 'log2'],
            'n_estimators': [200],
            'n_jobs': [1],
            'random_state': [42],
            'class_weight': ['balanced'],
        }),
        metric=roc_auc_score,
        greater_is_better=True,
        scoreLabel='roc-auc',
        n_jobs=-1,
        **kwargs,
    )
    return best_model, best_score
