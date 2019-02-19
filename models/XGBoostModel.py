from models.Model import Model
from xgboost.sklearn import XGBClassifier


class XGBoost(Model):
    def __init__(self):
        super().__init__()
        self.name = 'XGBoost'

        # SET MODEL
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
        self.param_grid = {'colsample_bytree': [0.6], 'gamma': [1], 'max_depth': [5], 'min_child_weight': [1], 'subsample': [1.0]}
        self.clf_raw = XGBClassifier(**ind_params)