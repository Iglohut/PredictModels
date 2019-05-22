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


        self.param_grid = {'colsample_bytree': [0.6], 'gamma': [1], 'max_depth': [5], 'min_child_weight': [1], 'subsample': [1.0],  'n_estimators':  [500]}

        # self.param_grid = {'colsample_bytree': [0.5, .75, 1.0],
        #                    'gamma': [0.5, 0.75, 1.0],
        #                    'max_depth': [3, 6, 10],
        #                    'subsample': [0.5, 0.8, 1.0],
        #                    'min_child_weight': [1, 2],
        #                    'eta': [0.01, 0.1, 0.2, 0.3, 0.4],
        #                    'lambda': [0.5, 0.8, 1.0],
        #                    'objective': ['binary:logistic'],
        #                    'n_estimators': [100],
        #                    'seed': [42],
        #                    'learning_rate': [0.1, 0.2, 0.3]
        #                    }


        # self.param_grid = {'colsample_bytree': [0.7],
        #                    'gamma': [0.5],
        #                    'max_depth': [10],
        #                    'subsample': [0.8],
        #                    'min_child_weight': [1],
        #                    'eta': [0.01, 0.1],
        #                    'lambda': [0.5],
        #                    'objective': ['binary:logistic'],
        #                    'n_estimators': [100, 500],
        #                    'seed': [42],
        #                    'learning_rate': [0.1]
        #                    }

        self.clf_raw = XGBClassifier()