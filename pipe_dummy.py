from category_encoders import OneHotEncoder


def tuples_encoder(cols):
    pipe = [
        (cols, OneHotEncoder(use_cat_names=True, impute_missing=False, drop_invariant=True))
    ]
    return pipe


if __name__ == '__main__':
    import pandas as pd
    df = pd.DataFrame({
        'cat1': ['a', 'a', 'a', 'b', None],
        'cat2': ['a', 'c', 'a', 'b', None]
    })
    from sklearn_pandas import DataFrameMapper
    pipe = DataFrameMapper([
        *tuples_encoder(['cat1', 'cat2'])
    ], df_out=True)
    print(pipe.fit_transform(df))


