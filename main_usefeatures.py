import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from sklearn.model_selection import train_test_split
from featureImportance import *
from testPerformance.testAUROC import get_auroc, test_auroc
from input_output.Saver import mySaver
from input_output.Loader import myLoader
from auxiliary.modelPlots import plotModelCorrelation, compareModelAcc
from auxiliary.funcs import flatten
from models.BayesModel import Bayes
from models.XGBoostModel import XGBoost
from models.MLPModel import MLP
from models.EnsembleRF import EnsembleRF
from models.EnsembleXG import EnsembleXG
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import StratifiedKFold
from models.RFmodel import RF
from models.Model import EnsembleModel
from auxiliary.featurePlots import StatsPlot, plot_topfeatures
from auxiliary.importData import ImportData
warnings.filterwarnings("ignore", category=DeprecationWarning)
import sys
# sys.setrecursionlimit(1500)

subsets = [None, 'or', 'od', 'con']
# For saving plots name
# figstring = str(subsets[0])
# condition = subsets[0]


for condition in subsets:
       figstring = str(condition)
       print("Doing condition: {}".format(figstring))

       # import the data
       Data = ImportData(condition=condition, remove_outliers=None)
       X = Data.X
       y = Data.y


       # Train and test data
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

       # Set stuff
       saver = mySaver()
       loader = myLoader()

       models = [RF, XGBoost]
       models = [m() for m in models]

       ensembleModels = [EnsembleRF, EnsembleXG]
       # ensembleModels = [m(models) for m in ensembleModels]

       for model in models:
              print("\nUsing ", model.name)

              # Train
              model.feature_selection(X_train)
              model.train(X_train, y_train)

              # Predict
              print("Predicting test set..")
              model.test(X_test, y_test)
              saver.save_predictions(model.predictions, 'predictions/' + model.name + '.csv')

              # Test significance of prediction
              model.get_pvalue_metric(X_test, y_test)

              # Computing Feature importances
              model.feature_importances(X_train, y_train, X_test, y_test, n_sim=None, relative=True)

              # Validate
              model.acc = test_auroc(model, pos_label=1, subset=figstring)
              print("AUROC on test set is:", model.acc)


       # # The ensemble models
       # for ensembleModel in ensembleModels:
       #        ensembleModel.feature_selection()
       #        ensembleModel.train(X_train, y_train)
       #        ensembleModel.test(X_test, y_test)
       #        saver.save_predictions(ensembleModel.predictions, 'predictions/' + ensembleModel.name + '.csv')
       #
       #        # Test significance of prediction
       #        ensembleModel.get_pvalue_metric(X_test, y_test)
       #
       #        # Feature importance
       #        ensembleModel.feature_importances(X_train, y_train, X_test, y_test)
       #
       #        # Validate
       #        ensembleModel.acc = test_auroc(ensembleModel.name)
       #        print("AUROC on test set is:", test_auroc(ensembleModel.name))




       # allModels = flatten([models, ensembleModels])
       allModels = models
       # Save models
       test_accuracies = [model.acc for model in allModels]
       saver.save_models(allModels, test_accuracies)

       # Compare models
       compareModelAcc(allModels, figname=figstring)
       plotModelCorrelation(allModels, figname=figstring)
       plot_featureimportances_drop(models, figname=figstring)
       plot_topfeatures(models, condition=condition, ntop=5)

       plt.close('all')
