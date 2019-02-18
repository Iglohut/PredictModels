# Base object, which can be used as a model for any task
# The model contains the feature selection for this model, the training and test methods
class Model(object):
    def __init__(self):
        self.params = params
        self.featureList = []
        self.acc = -1
        # NOTE: change this to the name of your model, it is used for the name of the prediction output file
        self.name = "baseModel"
        self.p_value = np.nan

    def feature_selection(self, X_train):
        pass

    # Training data should probably be a split of features and labels [X, Y]
    def train(self, X_train, y_train):
        pass

    def test(self, X_test, labels):
        pass

    def predict(self, X):
        pass