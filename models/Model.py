# Base object, which can be used as a model for any task
# The model contains the feature selection for this model, the training and test methods
from sklearn.ensemble import RandomForestClassifier # For example purposes
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score
from rfpimp import * # Feature importances permutation tests/drop-col
import warnings
from featureImportance import *
from testPerformance.testAUROC import get_auroc
from auxiliary.importData import ImportData

class Model(object):
    def __init__(self):
        self.featureList = []
        self.train_set_size = -1
        self.name = "RF"
        self.predictions =[]
        self.p_value = np.nan
        self.acc = np.nan

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


    def feature_selection(self, X_train, reselect=False):
        self.featureList = ImportData.features


        # This deletes all features that negatively influenced auroc metric
        if reselect:
            indices = self.featureImportances["Importances"] >= 0
            self.featureList = list(self.featureImportances["Features"][indices])
        self.featureList = list(X_train[self.featureList].dtypes[X_train[self.featureList].dtypes != 'object'].index)


    def feature_importances(self, X_train, y_train, X_test, y_test, n_sim = None, relative=False):
        "Computes feature importances base don drop-col: the ultimate measure."
        X_test = self._convertX(X_test)
        X_train = self._convertX(X_train)

        y_train = np.array(y_train).ravel()
        y_test = np.array(y_test).ravel()
        imp = dropcol_importances(self.clf.best_estimator_, X_train, y_train,X_test, y_test, metric=get_auroc)
        featureList = np.asarray(imp["Importance"]._stat_axis)
        featureImportances = np.array(imp["Importance"]._values)
        if relative:
            featureImportances = (featureImportances - featureImportances.min()) / (featureImportances - featureImportances.min()).sum() # Make relative
        self.featureImportances = {'Features': featureList,
                              'Importances': featureImportances,
                              'p_values': np.ones(len(featureList)),
                              }

        # If calculate p_value using permuation
        if n_sim is not None:
            print("Calculating p_values for feature importances...")
            permuation_importances = permutation_FI_list(self, X_train, y_train, X_test, y_test, self.featureImportances['Features'], n_sim=n_sim)

            if relative:
                # Normalize on ranking lowest 0, sum to 1..
                permuation_importances = (permuation_importances - permuation_importances.min()) / (permuation_importances - permuation_importances.min()).sum()

            p_values = [sum((permuation_importances[fi, :] > self.featureImportances["Importances"][fi])) / n_sim for fi
                        in range(len(self.featureImportances['Features']))]

            self.featureImportances["p_values"] = np.array(p_values)


    # Training data should probably be a split of features and labels [X, Y]
    def train(self, X_train, y_train):
        if self.featureList == []:
            raise ValueError('No features selected. Please first run feature selection.')
        X_train = self._convertX(X_train)
        # save trainingset size, prepare the data, and select the features
        self.train_set_size = len(X_train)
        X_train = np.array(X_train)
        y_train = np.array(y_train).ravel()

        print("Training model..")

        # Specific stuff here
        self.clf = GridSearchCV(self.clf_raw, param_grid=self.param_grid, cv=10, scoring="roc_auc", n_jobs=1)
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
        X_test = self._convertX(X_test)
        labels = np.array(labels)
        X_test = np.array(X_test)
        y_pred = self.clf.predict(X_test)

        # Save predictions (write to csv file later)
        self.predictions = []
        for i, prediction in enumerate(y_pred):
            self.predictions.append([labels[i][0], prediction])

    def get_pvalue_metric(self,X_test, y_test):
        "Computes permutation p-value for the roc_auc metric."
        X_test = self._convertX(X_test)
        cv = StratifiedKFold(2)
        score, permutation_scores, pvalue = permutation_test_score(
            self.clf.best_estimator_, X_test, np.ravel(y_test), scoring="roc_auc", cv=cv, n_permutations=100, n_jobs=1)
        self.p_value = pvalue
        print(self.name + " p-value roc_auc:", str(pvalue))

    def predict(self, X):
        if self.featureList == []:
            raise ValueError('No features selected. Please first run feature selection.')
        if self.train_set_size == -1:
            raise ValueError("Couldn't determine training set size, did you run feature_selection and train first?")
        X = self._convertX(X)
        X = np.array(X)
        return self.clf.predict(X)

    def _convertX(self, X):
        return X[self.featureList]

    def __repr__(self):
        return "Model({}, params: {})".format(self.name, self.param_grid)

    def __str__(self):
        return "Model({}, roc_auc: {:.2},p: {:.4})".format(self.name, self.acc, self.p_value)

    def __add__(self, other):
        warnings.warn("Warning.....Don't be adding models, use an ensemble model!")

class EnsembleModel(Model):
    def __init__(self, models):
        super().__init__()
        self.models = models
        # NOTE: change this to the name of your model, it is used for the name of the prediction output file
        self.name = "EnsemblebaseModel"

        # # SET MODEL: Set the classifier AND parameters to be used for train for each subcass

    def feature_selection(self):
        self.featureList = [model.name for model in self.models]

    def _convertX(self, X):
        if self.featureList == []:
            raise ValueError('No features selected. Please first run feature selection.')
        predictions = pd.DataFrame()
        for model in self.models:
            y_pred = model.predict(X)
            predictions = pd.concat([predictions, pd.DataFrame({model.name: y_pred})], axis=1, sort=False)
        return predictions[self.featureList]