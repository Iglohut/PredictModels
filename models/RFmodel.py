from models.Model import Model
from sklearn.ensemble import RandomForestClassifier

class RF(Model):
    def __init__(self):
        super().__init__()
        self.name = 'RF'

        # SET MODEL
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
