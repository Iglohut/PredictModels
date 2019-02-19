# Base object, which can be used as a model for any task
# The model contains the feature selection for this model, the training and test methods
from sklearn.ensemble import RandomForestClassifier # For example purposes
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score
from rfpimp import * # Feature importances permutation tests/drop-col

class Model(object):
    def __init__(self):
        self.featureList = []
        self.train_set_size = -1
        self.name = "RF"
        self.predictions =[]
        self.p_value = np.nan

        # NOTE: change this to the name of your model, it is used for the name of the prediction output file
        self.name = "baseModel"


        # SET MODEL: Set the classifier AND parameters to be used for train for each subcass
        ## EXAMPLE
        self.clf_raw = RandomForestClassifier()
        # param_grid = {'max_features': [1, int(np.sqrt(len(self.featureList))), len(self.featureList)],
        #               'max_depth': [3, None],
        #               'min_samples_split' :[2, 3, 10],
        #               'min_samples_leaf' : [1, 3, 10],
        #               'criterion':['gini', 'entropy'],
        #               'bootstrap':[True, False]}

        # best model so far
        self.param_grid = {'max_features': [4],
                      'max_depth': [None],
                      'min_samples_split' :[10],
                      'min_samples_leaf' : [10],
                      'criterion':['gini'],
                      'bootstrap':[True]}


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
        "Computes feature importances base don drop-col: the ultimate measure."
        y_train = np.array(y_train).ravel()
        y_test = np.array(y_test).ravel()
        imp = dropcol_importances(self.clf.best_estimator_, X_train, y_train,X_test, y_test)
        featureList = np.asarray(imp["Importance"]._stat_axis)
        featureImportances = np.array(imp["Importance"]._values)
        self.featureImportances = [featureList, featureImportances]

    # Training data should probably be a split of features and labels [X, Y]
    def train(self, X_train, y_train):
        if self.featureList == []:
            raise ValueError('No features selected. Please first run feature selection.')

        # save trainingset size, prepare the data, and select the features
        self.train_set_size = len(X_train)
        X_train = np.array(X_train[self.featureList])
        y_train = np.array(y_train).ravel()

        print("Training model..")

        # Specific stuff here
        self.clf = GridSearchCV(self.clf_raw, param_grid=self.param_grid, cv=10, scoring="roc_auc", n_jobs=2)
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
            raise ValueError("Couldn't determine training set size, did you run feature_selection and train first?")

        labels = np.array(labels)
        X_test = np.array(X_test[self.featureList])
        y_pred = self.clf.predict(X_test)

        # Save predictions (write to csv file later)
        self.predictions = []
        for i, prediction in enumerate(y_pred):
            self.predictions.append([labels[i][0], prediction])

    def get_pvalue_metric(self,X_test, y_test):
        "Computes permutation p-value for the roc_auc metric."
        cv = StratifiedKFold(2)
        score, permutation_scores, pvalue = permutation_test_score(
            self.clf.best_estimator_, X_test, np.ravel(y_test), scoring="roc_auc", cv=cv, n_permutations=100, n_jobs=1)
        self.p_value = np.copy(pvalue)
        print(self.name + " p-value roc_auc:", str(pvalue))

    def predict(self, X):
        if self.featureList == []:
            raise ValueError('No features selected. Please first run feature selection.')
        if self.train_set_size == -1:
            raise ValueError("Couldn't determine training set size, did you run feature_selection and train first?")
        X = np.array(X[self.featureList])
        return self.clf.predict(X)