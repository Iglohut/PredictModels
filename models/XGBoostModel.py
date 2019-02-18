from models.Model import Model
from sklearn.model_selection import GridSearchCV
import numpy as np
from xgboost.sklearn import XGBClassifier
from rfpimp import * # Feature importaces permutation tests

# This model contains the code for an ZGBoost model for the Titanic task, including
# feature selection, training and testing methods.
class XGBoost(object):
    def __init__(self):
        self.train_set_size = -1
        self.predictions =[]
        self.name = "XGBoost"
        self.p_value = np.nan

    def feature_selection(self, X_train):
        self.featureList = ['first_object', 'first_object_latency',
                            'stay1', 'stay2', 'SS1', 'perseverance', 'n_transitions',
                            'min1_n_explore', 'min2_n_explore', 'min3_n_explore', 'min4_n_explore',
                            'min5_n_explore', 'min1_obj1_time', 'min2_obj1_time', 'min3_obj1_time',
                            'min4_obj1_time', 'min5_obj1_time', 'min1_obj2_time', 'min2_obj2_time',
                            'min3_obj2_time', 'min4_obj2_time', 'min5_obj2_time', 'min1_DI',
                            'min2_DI', 'min3_DI', 'min4_DI', 'min5_DI']

        self.featureList = list(X_train[self.featureList].dtypes[X_train[self.featureList].dtypes != 'object'].index)

    def feature_importances(self, X_train, y_train, X_test, y_test):
        y_train = np.array(y_train).ravel()
        y_test = np.array(y_test).ravel()
        imp = dropcol_importances(self.clf.best_estimator_, X_train, y_train,X_test, y_test)
        featureList = np.asarray(imp["Importance"]._stat_axis)
        featureImportances = np.array(imp["Importance"]._values)
        self.featureImportances = [featureList, featureImportances]


    # train the model with the features determined in feature_selection()
    def train(self, X_train, y_train):
        if self.featureList == []:
            raise ValueError('No features selected. Please first run feature selection.')

        # save trainingset size, prepare the data, and select the features
        self.train_set_size = len(X_train)
        X_train = np.array(X_train[self.featureList])
        y_train = np.array(y_train).ravel()

        print("Training model..")

        # Hyper-parameter tuning
        # param_grid = {
        #     'min_child_weight': [1, 3],
        #     'gamma': [0.5, 1],
        #     'subsample': [0.8, 1.0],
        #     'colsample_bytree': [0.6, 0.8, 1.0],
        #     'max_depth': [3, 5]
        #     }
        # ind_params = {
        #     'learning_rate': 0.1,  # TODO find optimum
        #     'n_estimators': 1000,  # TODO find optimum
        #     'seed': 0,
        #     'subsample': 0.8,
        #     'objective': 'binary:logistic'
        # }

        ind_params = {'colsample_bytree': 0.6, 'gamma': 1, 'max_depth': 5, 'min_child_weight': 1, 'subsample': 1.0}
        param_grid = {'colsample_bytree': [0.6], 'gamma': [1], 'max_depth': [5], 'min_child_weight': [1], 'subsample': [1.0]}
        clf_raw = XGBClassifier(**ind_params)
        # param_grid = {'gamma': [0]}
        # clf_raw = XGBClassifier()


        # find best parameters
        self.clf = GridSearchCV(clf_raw, param_grid=param_grid, cv=10, scoring="roc_auc", n_jobs=2, verbose=1)
        self.clf.fit(X_train, y_train)

        print("Best parameters:")
        print(self.clf.best_params_)

        # print best performance of best model of gridsearch with cv
        self.acc = self.clf.best_score_
        print("Model with best parameters, train set avg CV accuracy:", self.acc)

    # predict the test set
    def test(self, X_test, labels):
        if self.featureList == []:
            raise ValueError('No features selected. Please first run feature selection.')
        if self.train_set_size == -1:
            raise ValueError("Couldn't determine training set size, did you run feature_selection and train first?")
        labels = np.array(labels)
        X_test = np.array(X_test[self.featureList])
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
        X = np.array(X[self.featureList])
        return self.clf.predict(X)