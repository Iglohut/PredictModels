import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

class MLP(object):
    def __init__(self):
        self.train_set_size = -1
        self.name = "MLP"
        self.predictions = []

    def feature_selection(self, X_train):
        self.featureList = ['first_object', 'first_object_latency',
                            'stay1', 'stay2', 'SS1', 'perseverance', 'n_transitions',
                            'min1_n_explore', 'min2_n_explore', 'min3_n_explore', 'min4_n_explore',
                            'min5_n_explore', 'min1_obj1_time', 'min2_obj1_time', 'min3_obj1_time',
                            'min4_obj1_time', 'min5_obj1_time', 'min1_obj2_time', 'min2_obj2_time',
                            'min3_obj2_time', 'min4_obj2_time', 'min5_obj2_time', 'min1_DI',
                            'min2_DI', 'min3_DI', 'min4_DI', 'min5_DI']

        self.featureList = list(X_train[self.featureList].dtypes[X_train[self.featureList].dtypes != 'object'].index)

    # Training data should probably be a split of features and labels [X, Y]
    def train(self, X_train, y_train):
        if self.featureList == []:
            raise ValueError('No features selected. Please first run feature selection.')

        # save trainingset size, prepare the data, and select the features
        self.train_set_size = len(X_train)
        X_train = np.array(X_train[self.featureList])
        y_train = np.array(y_train).ravel()

        print("Training model..")
        # param_grid = {'hidden_layer_sizes': [(5,2),(10,2),(5,3,2), (10,5,3,2)],
        #               'activation': ['identity', 'logistic', 'tanh', 'relu'],
        #               'solver': ['lbfgs', 'sgd', 'adam'],
        #               'learning_rate': ['constant', 'invscaling', 'adaptive']
        #               }

        param_grid = {'activation': ['identity'], 'hidden_layer_sizes': [(5, 3, 2)], 'learning_rate': ['constant'], 'solver': ['lbfgs']}

        clf_raw = MLPClassifier(random_state=1)
        # clf_raw = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (5, 2), random_state = 1)
        self.clf = GridSearchCV(clf_raw, param_grid=param_grid, cv=10, scoring="roc_auc", n_jobs=2, verbose=0)

        self.clf.fit(X_train, y_train)
        print("Best parameters:")
        print (self.clf.best_params_)
        self.acc = self.clf.best_score_
        print("Model with best parameters, train set avg CV accuracy:", self.acc)


    def test(self, X_test, labels):
        if self.featureList == []:
            raise ValueError('No features selected. Please first run feature selection.')
        if self.train_set_size == -1:
            raise ValueError("Couldn't determine training set size, did you run feature_selection and train first?")

        labels = np.array(labels)
        X_test = np.array(X_test[self.featureList])
        y_pred = self.clf.predict(X_test)

        # Save predictions (write to csv file later)
        self.predictions = []
        for i, prediction in enumerate(y_pred):
            self.predictions.append([labels[i][0], prediction])


    def predict(self, X):
        if self.featureList == []:
            raise ValueError('No features selected. Please first run feature selection.')
        if self.train_set_size == -1:
            raise ValueError("Couldn't determine training set size, did you run feature_selection and train first?")
        X = np.array(X[self.featureList])
        return self.clf.predict(X)