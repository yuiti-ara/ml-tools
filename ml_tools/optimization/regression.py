import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args


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
            scores = cross_val_score(model_xg, X_tr, y_tr, cv=4, scoring="neg_mean_absolute_error")
            score = -np.mean(scores)
        else:
            model_xg.fit(X_tr, y_tr)
            y_vl_pred = model_xg.predict(X_vl)
            score = mean_absolute_error(y_vl, y_vl_pred)

        return score

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
    space = [
        Categorical([10, 20, 30, 40, 50, 60, 70, 80, 100], name='min_samples_leaf'),
        Categorical([1, .75, .5, 'sqrt', 'log2'], name='max_features'),
    ]

    @use_named_args(space)
    def objective(**params):
        model_rf.set_params(**params)

        if X_vl is None or y_vl is None:
            scores = cross_val_score(model_rf, X_tr, y_tr, cv=4, scoring="neg_mean_absolute_error")
            score = -np.mean(scores)
        else:
            model_rf.fit(X_tr, y_tr)
            y_vl_pred = model_rf.predict(X_vl)
            score = mean_absolute_error(y_vl, y_vl_pred)

        return score

    resp = gp_minimize(
        objective,
        space,
        n_calls=100,
        random_state=42,
        verbose=True,
        n_jobs=-1,
    )
    return resp
