from models.Model import Model
from sklearn.neural_network import MLPClassifier

class MLP(Model):
    def __init__(self):
        super().__init__()
        self.name = 'MLP'

        # SET MODEL
        # param_grid = {'hidden_layer_sizes': [(5,2),(10,2),(5,3,2), (10,5,3,2)],
        #               'activation': ['identity', 'logistic', 'tanh', 'relu'],
        #               'solver': ['lbfgs', 'sgd', 'adam'],
        #               'learning_rate': ['constant', 'invscaling', 'adaptive']
        #               }

        self.param_grid = {'activation': ['identity'], 'hidden_layer_sizes': [(5, 3, 2)], 'learning_rate': ['constant'],
                      'solver': ['lbfgs']}

        self.clf_raw = MLPClassifier(random_state=1)
        # clf_raw = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (5, 2), random_state = 1)