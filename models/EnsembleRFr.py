import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

class EnsembleRFr(object):
    def __init__(self, models):
        self.train_set_size = -1
        self.name = "EnsembleRFr"
        self.predictions =[]
        self.models = models
        self.p_value = np.nan

    def feature_selection(self):
        self.featureList = [model.name for model in self.models]

    def train(self, X_train, y_train):
        if self.featureList == []:
            raise ValueError('No features selected. Please first run feature selection.')

        predictions = pd.DataFrame()
        for model in self.models:
            y_pred = model.predict(X_train)
            predictions = pd.concat([predictions, pd.DataFrame({model.name: y_pred})], axis=1, sort=False)

        self.train_set_size = len(X_train)
        X_train = np.array(predictions)
        y_train = np.array(y_train).ravel()
        clf_raw = RandomForestRegressor()

        param_grid = {'max_features': [4],
                      'max_depth': [None],
                      'min_samples_split' :[10],
                      'min_samples_leaf' : [10],
                      'criterion':['mse', 'mae'],
                      'bootstrap':[True]}

        # find best parameters
        self.clf = GridSearchCV(clf_raw, param_grid=param_grid, cv=10, scoring="roc_auc", n_jobs=2)
        self.clf.fit(X_train, y_train)

        print("Best parameters:")
        print(self.clf.best_params_)

        # print best performance of best model of gridsearch with cv
        self.acc = self.clf.best_score_
        print("Model with best parameters, train set avg CV accuracy:", self.acc)

    def test(self, X_test, labels):
        if self.featureList == []:
            raise ValueError('No features selected. Please first run feature selection.')
        if self.train_set_size == -1:
            raise ValueError(
                "Couldn't determine training set size, did you run feature_selection and train first?")

        predictions = pd.DataFrame()
        for model in self.models:
            y_pred = model.predict(X_test)
            predictions = pd.concat([predictions, pd.DataFrame({model.name: y_pred})], axis=1, sort=False)

        X_test = np.array(predictions)

        labels = np.array(labels)
        y_pred = self.clf.predict(X_test)

        # Save predictions (write to csv file later)
        self.predictions = []
        for i, prediction in enumerate(y_pred):
            self.predictions.append([labels[i][0], prediction])



    def predict(self, X):
        if self.featureList == []:
            raise ValueError('No features selected. Please first run feature selection.')
        if self.train_set_size == -1:
            raise ValueError("Couldn't determine training set size, did you run feature_selection and train first?")

        predictions = pd.DataFrame()
        for model in self.models:
            y_pred = model.predict(X)
            predictions = pd.concat([predictions, pd.DataFrame({model.name: y_pred})], axis=1, sort=False)


        X = np.array(predictions)
        return self.clf.predict(X)