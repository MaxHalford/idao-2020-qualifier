import datetime as dt

import joblib
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing


class Pipeline:
    """Barebones implementation with less overhead than sklearn."""

    def __init__(self, *steps):
        self.steps = steps

    def fit(self, X, y):
        for transformer in self.steps[:-1]:
            X = transformer.fit_transform(X, y)
        self.steps[-1].fit(X, y)
        return self

    def predict(self, X):
        for transformer in self.steps[:-1]:
            X = transformer.transform(X)
        return self.steps[-1].predict(X)


class StandardScaler(preprocessing.StandardScaler):
    """Barebones implementation with less overhead than sklearn."""

    def transform(self, X):
        return (X - self.mean_) / self.var_ ** .5


class LinearRegression(linear_model.LinearRegression):
    """Barebones implementation with less overhead than sklearn."""

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_


class ARModel:

    def __init__(self, p, model):
        self.p = p
        self.model = model

    def fit(self, path):

        n = path.strides[0]
        X = np.lib.stride_tricks.as_strided(path, shape=(path.shape[0], self.p), strides=(n, n))[:-self.p]
        Y = path[self.p:]

        # Save the most recent history for later
        self.history = path[-self.p:].reshape(1, -1)

        self.model.fit(X, Y)

    def forecast(self, steps):

        history = self.history.copy()
        predictions = np.empty(steps)

        for i in range(steps):

            y_pred = self.model.predict(history)[0]
            predictions[i] = y_pred

            # Shift forward (faster than np.roll)
            history[0, :-1] = history[0, 1:]
            history[0, -1] = y_pred

        return predictions


if __name__ == '__main__':

    models = joblib.load('ar_models.pkl')

    test = pd.read_csv(
        'test.csv',
        #'../data/Track 1/test.csv',
        parse_dates=['epoch'],
        usecols=['id', 'sat_id', 'epoch']
    )
    preds = []

    for sat, g in test.groupby('sat_id'):

        sat_models = models[sat]

        for var in ('x', 'y', 'z', 'Vx', 'Vy', 'Vz'):

            model = sat_models[var]
            pred = model.forecast(len(g)).astype('float32')

            preds.append(pd.DataFrame({
                'id': g['id'],
                'sat_id': sat,
                'epoch': g['epoch'],
                'y_pred': pred,
                'variable': var
            }))

    preds = pd.concat(preds)
    preds = preds.groupby('sat_id').apply(lambda g: g.pivot_table(index=['id', 'epoch'], columns='variable', values='y_pred')).reset_index()

    correct_preds = []

    cols_to_shift = ['x', 'y', 'z', 'Vx', 'Vy', 'Vz']

    for _, g in preds.groupby('sat_id'):

        g = g.copy()
        dups = g[g['epoch'].diff() < dt.timedelta(seconds=60)].index

        for i in dups:
            g.loc[i:, cols_to_shift] = g.loc[i:, cols_to_shift].shift()
        g[cols_to_shift] = g[cols_to_shift].ffill()

        correct_preds.append(g)

    correct_preds = pd.concat(correct_preds)
    correct_preds[['id', 'x', 'y', 'z', 'Vx', 'Vy', 'Vz']].to_csv('submission.csv', index=False)
