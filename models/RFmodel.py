from models.Model import Model
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RF(Model):
    def __init__(self):
        super().__init__()
        self.name = 'RF'

        # SET MODEL
        self.clf_raw = RandomForestClassifier()
        # self.param_grid = {'max_features': [0.1, 0.25, 0.5, 0.75, 1],
        #               'max_depth': [1, 3, 7, None],
        #               'min_samples_split' :[2, 3, 10],
        #               'min_samples_leaf' : [1, 3, 10],
        #               'criterion':['gini', 'entropy'],
        #               'bootstrap':[True, False],
        #               'n_estimators': [5, 10, 15, 20]}

        # best model so far
        # self.param_grid = {'max_features': [4],
        #               'max_depth': [None],
        #               'min_samples_split' :[10],
        #               'min_samples_leaf' : [10],
        #               'criterion':['gini'],
        #               'bootstrap':[True]}

        self.param_grid = {'bootstrap': [True],
                           'criterion': ['entropy', 'gini'],
                           'max_depth': [None],
                           'max_features': [0.1, 0.25],
                           'min_samples_leaf': [1],
                           'min_samples_split': [10],
                           'n_estimators': [500]}
