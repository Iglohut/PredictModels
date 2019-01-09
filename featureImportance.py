import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from rfpimp import *
import matplotlib.pyplot as plt
import seaborn as sns

# analyze the feature importance in a random forest model
# see: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
def analyze_feature_importance(forest, feature_labels):
    # This is the GINI importance
    # obtain relative feature importances
    importances = forest.feature_importances_
    # compute standard deviation tree-wise
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    # get the feature indices
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(len(feature_labels)):
        print("%d. feature %s (%f)" % (f + 1, feature_labels[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    fig, ax = plt.subplots(1, 1)
    plt.title("Feature importances")
    plt.bar(range(len(feature_labels)), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(len(feature_labels)), indices)
    list(np.array(feature_labels)[indices])
    ax.set_xticklabels(list(np.array(feature_labels)[indices]), rotation='vertical')
    plt.xlim([-1, len(feature_labels)])
    plt.show()
    plt.tight_layout()

def analyze_feature_importance2(forest, X_test, y_test):
    imp = importances(forest, X_test, y_test)  # permutation
    plt.figure()
    viz = plot_importances(imp)
    viz.view()


def obtain_feature_scores(scoring_metric,x_data,y_data,feature_labels):
    rating = scoring_metric(x_data,y_data)
    if isinstance(rating,tuple):
        rating, _ = rating
    indices = np.argsort(rating)
    # invert
    indices = indices[::-1]
    for f in range(len(feature_labels)):
        print("%d. feature %s (%f)" % (f + 1, feature_labels[indices[f]], rating[indices[f]]))

    return rating





def analyze_feature_importances_all(models):
    posModels = []
    for model in models:
        try:
            model.clf.best_estimator_.feature_importances_
            posModels.append(model)
        except:
            pass

    ncols = int(np.ceil(np.sqrt(len(posModels))))
    nrows = int(np.round(np.sqrt(len(posModels))))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex="all", figsize=(15, 15))

    names_classifiers = [(model.name, model) for model in posModels]

    nclassifier = 0
    for row in range(nrows):
        for col in range(ncols):
            if nclassifier <= len(posModels):
                name = names_classifiers[nclassifier][0]
                classifier = names_classifiers[nclassifier][1]
                indices = np.argsort(classifier.clf.best_estimator_.feature_importances_)[::-1][:40]
                g = sns.barplot(y=np.array(classifier.featureList)[indices][:40],
                                x=classifier.clf.best_estimator_.feature_importances_[indices][:40], orient='h',
                                ax=axes[row][col])
                g.set_xlabel("Relative importance", fontsize=12)
                g.set_ylabel("Features", fontsize=12)
                g.tick_params(labelsize=9)
                g.set_title(name + " feature importance")
                nclassifier += 1

    plt.show()
    plt.tight_layout()
    plt.savefig('./figs/FeatureImportances.pdf')