from sklearn import metrics
import pandas as pd
import numpy as np
import os

def test_auroces(modelNames, X, y):
    return [get_auroc(model,X, y) for model in modelNames]

def get_auroc(model, X, y, sample_weights = None, pos_label=1):
    # y_pred = model.predict(np.array(X))# If permutaton test
    y_pred = model.predict(X) # If dropcol test
    # y_pred = np.array(y_pred > 0.5, dtype=np.int)
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred, pos_label=pos_label)
    auroc = metrics.auc(fpr, tpr)
    return auroc




def test_auroc(model, pos_label=1, subset = None):
    modelName = model.name
    model_ = pd.read_csv('predictions/' + modelName + '.csv')
    y = model_["Gene"]
    y_pred = model_["PredictedGene"]
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred, pos_label=pos_label)
    auroc = metrics.auc(fpr, tpr)


    # Save stuff to plot later
    path_save = os.getcwd() + '/Data/' + subset + '_' + modelName + '_metrics.csv'
    data = {'condition':  subset,'model': modelName,'fpr': fpr, 'tpr': tpr, 'threshold': thresholds,  'AUROC': auroc, 'p_value': model.p_value}
    df = pd.DataFrame(data=data)
    df.to_csv(path_save, index=False)
    print("Saved ROC metrics to", path_save)
    return auroc

