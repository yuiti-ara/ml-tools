from rfpimp import importances


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
