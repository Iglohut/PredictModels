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
#
# ve = EnsembleRF(models)
# ve.feature_selection()
# ve.train(X_train, y_train)
# ve.test(X_test, y_test)
# saver.save_predictions(ve.predictions, 'predictions/' + ve.name + '.csv')
# print("AUROC on test set is:", test_auroc(ve.name))

[models.append(m) for m in ensembleModels]

# Save models
test_accuracies = [test_auroc(model.name) for model in models]
saver.save_models(models, test_accuracies)

# Compare models
compareModelAcc(models)
plotModelCorrelation(models)

analyze_feature_importances_all(models)
#
# models[2].clf.best_estimator_.feature_importances_
#
# analyze_feature_importance2(models[2].clf.best_estimator_, X_test, y_test)
#
#
# import matplotlib.pyplot as plt
# import seaborn as sns
# posModels = []
# for model in models:
#        try:
#         model.clf.best_estimator_.feature_importances_
#         posModels.append(model)
#        except:
#               pass
#
#
# ncols = int(np.ceil(np.sqrt(len(posModels))))
# nrows = int(np.round(np.sqrt(len(posModels))))
# fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,15))
#
# names_classifiers = [(model.name, model) for model in posModels]
#
# nclassifier = 0
# for row in range(nrows):
#     for col in range(ncols):
#         name = names_classifiers[nclassifier][0]
#         classifier = names_classifiers[nclassifier][1]
#         indices = np.argsort(classifier.clf.best_estimator_.feature_importances_)[::-1][:40]
#         g = sns.barplot(y=np.array(classifier.featureList)[indices][:40], x = classifier.clf.best_estimator_.feature_importances_[indices][:40] , orient='h',ax=axes[row][col])
#         g.set_xlabel("Relative importance",fontsize=12)
#         g.set_ylabel("Features",fontsize=12)
#         g.tick_params(labelsize=9)
#         g.set_title(name + " feature importance")
#         nclassifier += 1
#
# #
# ve = EnsembleRF(models)
# ve.feature_selection()
# ve.train(X_train, y_train)
# ve.test(X_test, y_test)
# saver.save_predictions(ve.predictions, 'predictions/' + ve.name + '.csv')
# print("AUROC on test set is:", test_auroc(ve.name))
#
#
# predictions = pd.DataFrame()
# predictions_test = pd.DataFrame()
# for model in models:
#        y_pred = model.predict(X_train)
#        df = pd.DataFrame({model.name: y_pred})
#        predictions = pd.concat([predictions, pd.DataFrame({model.name: y_pred})], axis=1, sort=False)
#
#        y_pred = model.predict(X_test)
#        df = pd.DataFrame({model.name: y_pred})
#        predictions_test = pd.concat([predictions_test, pd.DataFrame({model.name: y_pred})], axis=1, sort=False)
#
#
# X_test_ensemble = np.array(predictions_test)
# X_train_ensemble = np.array(predictions)
# clf_raw = RandomForestClassifier()
# clf_raw.fit(X_train_ensemble, y_train)
# auroc_train = get_auroc(clf_raw, X_train_ensemble, y_train)
# auroc_val = get_auroc(clf_raw, X_test_ensemble, y_test)
#
# features = [model.name for model in models]
# analyze_feature_importance(clf_raw, features)
# analyze_feature_importance2(clf_raw, predictions_test, y_test)
#
# # # Train model
# # train_set_size = len(X_train)
# # train_X = np.array(X_train[features])
# # train_Y = np.array(y_train)
# #
# # clf_raw = RandomForestClassifier(max_depth= None, max_features= 1, min_samples_leaf= 3, min_samples_split= 3, bootstrap=False, criterion='entropy')
# # # clf_raw = RandomForestRegressor(max_depth= None, max_features= 1, min_samples_leaf= 3, min_samples_split= 3, bootstrap=False)
# # clf_raw.fit(X_train, y_train)
# # #
# # # param_grid = {'max_features': [1, int(np.sqrt(len(features))), len(features)],
# # #               'max_depth': [3, None],
# # #               'min_samples_split' :[2, 3, 10],
# # #               'min_samples_leaf' : [1, 3, 10],
# # #               'criterion':['gini', 'entropy'],
# # #               'bootstrap':[True, False]}
# # #
# # # clf = GridSearchCV(clf_raw, param_grid=param_grid, cv=10, scoring='roc_auc' , n_jobs=2)
# # # clf.fit(train_X, train_Y)
# # #
# # # print("Best parameters:")
# # # print(clf.best_params_)
# # #
# # # clf.best_score_
# #
# #
# # # Test model
# # auroc_train = get_auroc(clf_raw, X_train, y_train)
# # auroc_val = get_auroc(clf_raw, X_test, y_test)
# #
# #
# # # Feature importances of RandomForest
# # analyze_feature_importance(clf_raw, features)
# # analyze_feature_importance2(clf_raw, X_test, y_test)
# #
