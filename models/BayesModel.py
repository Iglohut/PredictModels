from models.Model import Model
from sklearn.naive_bayes import GaussianNB

class Bayes(Model):
    def __init__(self):
        super().__init__()
        self.name = 'Bayes'

        # SET MODEL
        self.clf_raw = GaussianNB()
        self.param_grid = {'priors': [None]}