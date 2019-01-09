import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from featureImportance import analyze_feature_importance, obtain_feature_scores, analyze_feature_importance2, analyze_feature_importances_all
from testPerformance.testAUROC import get_auroc, test_auroc
from input_output.Saver import mySaver
from input_output.Loader import myLoader
from auxiliary.modelPlots import plotModelCorrelation, compareModelAcc
from models.RFmodel import RF
from models.RFrmodel import RFr
from models.BayesModel import Bayes
from models.XGBoostModel import XGBoost
from models.MLPModel import MLP
from models.EnsembleRF import EnsembleRF
from models.EnsembleXG import EnsembleXG


# Open file
file = './Data/SS_alldata_OS_ehmt1.csv'
df = pd.read_csv(file)

# General preprocessing
df = df.loc[~df.subject.isin([8, 9, 10, 11, 12])] # Don't go over round 2 subjects

fillFeatures = ['stay1', 'stay2', 'SS1'] # Features that could be nan if mouse never switched object
df[fillFeatures] = df[fillFeatures].fillna(-1) # Fill the nans with -1: choose more logical value?

df['condition'].replace(['con', 'od', 'or'], [0,1, 2], inplace=True)

df = df.dropna()

# # Select features
features = ['first_object', 'first_object_latency',
       'stay1', 'stay2', 'SS1', 'perseverance', 'n_transitions',
       'min1_n_explore', 'min2_n_explore', 'min3_n_explore', 'min4_n_explore',
       'min5_n_explore', 'min1_obj1_time', 'min2_obj1_time', 'min3_obj1_time',
       'min4_obj1_time', 'min5_obj1_time', 'min1_obj2_time', 'min2_obj2_time',
       'min3_obj2_time', 'min4_obj2_time', 'min5_obj2_time', 'min1_DI',
       'min2_DI', 'min3_DI', 'min4_DI', 'min5_DI']

target = ["genotype"]


# Make input and output
X = df[features]
y = df[target]

# Train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Set stuff
saver = mySaver()
loader = myLoader()

models = [RF, RFr, Bayes, XGBoost, MLP]
models = [m() for m in models]

ensembleModels = [EnsembleRF, EnsembleXG]
ensembleModels = [m(models) for m in ensembleModels]

for model in models:
       print("\nUsing ", model.name)

       # Train
       model.feature_selection(X_train)
       model.train(X_train, y_train)

       # Predict
       print("Predicting test set..")
       model.test(X_test, y_test)
       saver.save_predictions(model.predictions, 'predictions/' + model.name + '.csv')

       # Validate
       # model.acc = test_auroc(model.name)
       print("AUROC on test set is:", test_auroc(model.name))


# The ensemble model
for ensembleModel in ensembleModels:
       ensembleModel.feature_selection()
       ensembleModel.train(X_train, y_train)
       ensembleModel.test(X_test, y_test)
       saver.save_predictions(ensembleModel.predictions, 'predictions/' + ensembleModel.name + '.csv')
       print("AUROC on test set is:", test_auroc(ensembleModel.name))
[models.append(m) for m in ensembleModels]

# Save models
test_accuracies = [test_auroc(model.name) for model in models]
saver.save_models(models, test_accuracies)

# Compare models
compareModelAcc(models)
plotModelCorrelation(models)
analyze_feature_importances_all(models)