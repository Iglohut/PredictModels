from sklearn import metrics
import pandas as pd
import numpy as np

def test_auroces(modelNames, X, y):
    return [get_auroc(model,X, y) for model in modelNames]

def get_auroc(model, X, y, sample_weights = None, pos_label=1):
    # y_pred = model.predict(np.array(X))# If permutaton test
    y_pred = model.predict(X) # If dropcol test
    # y_pred = np.array(y_pred > 0.5, dtype=np.int)
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred, pos_label=pos_label)
    auroc = metrics.auc(fpr, tpr)
    return auroc




def test_auroc(modelName, pos_label=1):
    model = pd.read_csv('predictions/' + modelName + '.csv')
    y = model["Gene"]
    y_pred = model["PredictedGene"]
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred, pos_label=pos_label)
    auroc = metrics.auc(fpr, tpr)
    return auroc

