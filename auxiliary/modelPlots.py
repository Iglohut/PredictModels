import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
def compareModelAcc(models, figname=None):
    '''
    Receives as input a list models, containing model objects.
    Plots the accuracy of each model in a barplot.
    '''
    cv_means = []
    cv_std = []
    model_names = []
    p_values = []

    for model in models:
        if hasattr(model, 'acc'):
            model_names.append(model.name)
            cv_means.append(model.acc)
            # cv_std.append(cv_result.std())
            cv_std.append(0)
            p_values.append(model.p_value)

    cv_res = pd.DataFrame({"CrossValMeans":cv_means,"Pvalues": p_values,"Algorithm":model_names})
    plt.figure()
    g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, color="grey",orient = "h")
    g.set_xlabel("Mean Accuracy")
    g.set_title("Cross validation scores")
    p_values = np.array(p_values)
    p_strings = (p_values < 0.05).astype(int) + (p_values < 0.01) + (p_values < 0.001)

    for i, v in enumerate(p_strings):
        g.text(cv_means[i] + cv_means[i].max() * 0.02 , i, "".join(["*"] * v), color='black', ha="center")
    plt.show()
    plt.tight_layout()

    if figname is None:
        plt.savefig('./figs/modelAccs.pdf')
    else:
        savename = './figs/' + figname + '_modelAccs.pdf'
        plt.savefig(savename)



def plotModelCorrelation(models, figname=None):
    '''
    Receives as input a list models, containing model objects.
    Calculates the correlation between each model of the output predictions.
    The correlation here is defined as the overlap of the two prediction files.
    '''
    predictions = []

    # load only the predictions of each model and append to the list
    for model in models:
        df = pd.read_csv('predictions/' + model.name + '.csv')
        df = df.drop('Gene', axis=1)
        df.rename(columns = {'PredictedGene': model.name}, inplace = True)
        predictions.append(df)

    # concatenate and plot predictions
    ensemble_results = pd.concat(predictions, axis=1)
    plt.figure()
    g = sns.heatmap(overlap_correlation(ensemble_results), annot=True)
    # g = sns.heatmap(ensemble_results.corr(), annot=True)
    g = g.set_title("Overlap of generated predictions of models")
    plt.show()
    plt.tight_layout()

    plt.show()
    plt.tight_layout()
    if figname is None:
        plt.savefig('./figs/modelCorrs.pdf')
    else:
        savename = './figs/' + figname + 'modelCorrs.pdf'
        plt.savefig(savename)




def overlap_correlation(ensemble_results):
    '''
    Receives as input a n-d array, with the predictions of each model in a seperate column
    Calculates the overlap between the given arrays (containing predictions of a model).
    Thus: 2 models which have prediction arrays containing only 2 identical
    predictions at the same index, and 98 wrong ones, have an overlap of 0.02.
    '''
    numeric_df = ensemble_results._get_numeric_data()
    cols = numeric_df.columns
    idx = cols.copy()
    mat = numeric_df.values.T
    min_periods = 1
    corrf = overlap
    K = len(cols)
    correl = np.empty((K, K), dtype=float)
    mask = np.isfinite(mat)
    for i, ac in enumerate(mat):
        for j, bc in enumerate(mat):
            if i > j:
                continue
            valid = mask[i] & mask[j]
            if valid.sum() < min_periods:
                c = np.nan
            elif i == j:
                c = 1.
            elif not valid.all():
                c = corrf(ac[valid], bc[valid])
            else:
                c = corrf(ac, bc)
            correl[i, j] = c
            correl[j, i] = c


    correl = pd.DataFrame(correl, index=idx, columns=cols)
    return correl

# return howhow many of the items are identical in both arrays
def overlap(a, b):
    l1 = float(len(a))
    l2 = np.count_nonzero(np.array(a)==np.array(b))
    l = l2/l1
    return l