import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from models.Model import EnsembleModel


class EnsembleRF(EnsembleModel):
    def __init__(self, models):
        super().__init__(models)
        self.name = 'EnsembleRF'

        # SET MODEL
        self.feature_selection()
        self.clf_raw = RandomForestClassifier()

        self.param_grid = {'max_features': [1, int(np.sqrt(len(self.featureList))), len(self.featureList)-1],
                      'max_depth': [3, None],
                      'min_samples_split' :[2, 3, 10],
                      'min_samples_leaf' : [1, 3, 10],
                      'criterion':['gini', 'entropy'],
                      'bootstrap':[True, False]}