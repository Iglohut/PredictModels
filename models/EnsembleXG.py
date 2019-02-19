from models.Model import EnsembleModel
from xgboost.sklearn import XGBClassifier

class EnsembleXG(EnsembleModel):
    def __init__(self, models):
        super().__init__(models)
        self.name = 'EnsembleXG'

        # SET MODEL
        ind_params = {'colsample_bytree': 0.6, 'gamma': 1, 'max_depth': 5, 'min_child_weight': 1, 'subsample': 1.0}
        self.param_grid = {'colsample_bytree': [0.6], 'gamma': [1], 'max_depth': [5], 'min_child_weight': [1], 'subsample': [1.0]}
        self.clf_raw = XGBClassifier(**ind_params)