import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from rfpimp import *
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from multiprocessing import Pool
import psutil
from testPerformance.testAUROC import get_auroc
from auxiliary.funcs import flatten

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
            if nclassifier < len(posModels):
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


def plot_featureimportances_drop(models, figname=None):
    """
    Uses dropcolumn permuation
    :param models:
    :return:
    """
    ncols = int(np.ceil(np.sqrt(len(models))))
    nrows = int(np.round(np.sqrt(len(models))))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex="all", figsize=(15, 15))

    names_classifiers = [(model.name, model) for model in models]

    nclassifier = 0
    for row in range(nrows):
        for col in range(ncols):
            if nclassifier < len(models):
                name = names_classifiers[nclassifier][0]
                classifier = names_classifiers[nclassifier][1]
                indices = np.array(flatten(np.argsort(classifier.featureImportances['Importances'])[::-1][:40])) # Importacces

                if nrows > 1:
                    g = sns.barplot(y=classifier.featureImportances['Features'][indices][:40],  # Featurelist
                                    x=classifier.featureImportances['Importances'][indices][:40],color="grey", orient='h',
                                    ax=axes[row][col])
                else:
                    g = sns.barplot(y=classifier.featureImportances['Features'][indices][:40],  # Featurelist
                                    x=classifier.featureImportances['Importances'][indices][:40],color="grey", orient='h',
                                    ax=axes[col])

                g.set_xlabel("Relative importance", fontsize=12)
                g.set_ylabel("Features", fontsize=12)
                g.tick_params(labelsize=9)
                g.set_title(name + " feature importance")

                # Print p-values as asterixes
                p_values = np.array(classifier.featureImportances["p_values"][indices][:40])
                p_strings = (p_values < 0.05).astype(int) + (p_values < 0.01) + (p_values < 0.001)
                for i, v in enumerate(p_strings):
                    g.text(classifier.featureImportances['Importances'][indices][:40][i] + classifier.featureImportances['Importances'].max()*0.02, i+0.5, "".join(["*"] * v), color='black', ha="center")

                nclassifier += 1
    plt.show()
    plt.tight_layout()
    if figname is None:
        plt.savefig('./figs/FeatureImportances.pdf')
    else:
        savename = './figs/' +figname + '_FeatureImportances.pdf'
        plt.savefig(savename)



# Permutation test on features
class OnePerm:
    """
    Does one permutation on all the features for testing feature importances using drop-col method.
    """
    def __init__(self, model, X_train, y_train, X_test, y_test, featureList):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.featureList = featureList

    def __call__(self, i=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # for col in self.X_train.columns:
            #     self.X_train[col] = np.random.permutation(self.X_train[col])


            for col in self.X_test.columns:
                self.X_test[col] = np.random.permutation(self.X_test[col])

            # self.clf_tmp = clone(self.model.clf.best_estimator_)
            # self.clf_tmp.random_state = 999
            # self.clf_tmp.fit(self.X_train, self.y_train)

            # self.imp_tmp = dropcol_importances(self.clf_tmp, self.X_train, self.y_train, self.X_test, self.y_test, metric=get_auroc)

            self.imp_tmp = importances(self.model.clf.best_estimator_, self.X_test, self.y_test, metric=get_auroc)
        self.featureImportances_tmp = np.array(self.imp_tmp["Importance"][self.featureList]._values)



        return self.featureImportances_tmp


def permutation_FI_list(model, X_train, y_train, X_test, y_test, featureList, n_sim = 100):
    """
    :param model: full class model
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param featureList:
    :param n_sim:
    :return: list of n_sim permutations per feature
    """
    # Make class
    oneperm = OnePerm(model, X_train, y_train, X_test, y_test, featureList)

    # # Set CPU's ready
    # p = psutil.Process()
    # p.cpu_affinity()
    # all_cpus = list(range(psutil.cpu_count()))
    # p.cpu_affinity(all_cpus)
    #
    # # Multiprocessing
    # p = Pool(6)
    # out_list = p.map(oneperm, range(n_sim))
    # p.close()
    # p.join()
    #
    #
    # return np.array(out_list).T
    perms = [oneperm(i) for i in range(n_sim)]

    return perms
