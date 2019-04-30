import pandas as pd
import seaborn as sns
import numpy as np
from mlxtend.evaluate import permutation_test
import matplotlib.pyplot as plt

class StatsPlot:
    df = pd.read_csv('./Data/SS_alldata_OS_ehmt1.csv')
    def __init__(self, features, condition = None, nsim=10000):
        # Select condition to plot
        if condition is not None:
            self.df = self.df[self.df['condition'] == condition]

        # Set main vars
        self.features = features # features to plot as y-value
        self.nsim = nsim # number of simulations of permuation test

        # Select savename
        if condition is not None:
            self.savename = './figs/' + condition + '_topfeatureplots.pdf'
        else:
            self.savename = './figs/all_topfeatureplots.pdf'

        # Ploting: goal of this class
        self.plot()

    def plot(self):
        """
        Plots the selected features split by genotype and
        indicates if group differences are significant according to the permutation distribution,
        """
        n_features = len(self.features)

        # Set figure size
        maxcols = 5
        if n_features <= maxcols:
            ncols = n_features
            nrows = 1
        else:
            ncols = maxcols
            nrows =int(np.floor(n_features/ncols) + np.mod(n_features, ncols))
        f, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex="all", figsize=(15, 15))
        meanprops = {"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"}

        # Make subplots
        counter = 0
        for row in range(nrows):
            for col in range(ncols):
                if counter > (n_features - 1): break
                feature = self.features[counter]
                if nrows > 1:
                    g = sns.boxplot(x='genotype', y=feature, color="grey", data=self.df, ax=axes[row][col], order=range(2), showmeans=True, meanprops=meanprops)
                else:
                    g = sns.boxplot(x='genotype', y=feature, color="grey", data=self.df, ax=axes[col], order=range(2), showmeans=True, meanprops=meanprops)

                # Get statistical significance of difference of means
                p_value = self.ptest(feature)

                # Plot pvalue indication
                p_string = ((p_value < 0.05) + (p_value < 0.01) + (
                            p_value < 0.001)) * '*'  # To visually indicate pvalue
                if len(p_string) == 0: p_string = 'n.s.'  # if not significant, set string o n.s.
                x1, x2 = 0, 1
                y, h, col = self.df[feature].max() + 0.04 * self.df[feature].max(), 0.04 * self.df[feature].max(), 'k'
                g.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
                g.text((x1 + x2) * .5, y + h, p_string, ha='center', va='bottom', color=col)

                counter += 1
        plt.tight_layout()
        plt.savefig(self.savename)


    def ptest(self, feature):
        """
        :param feature: dependent variable
        :return: two-sided probability of group means differ under permutation distribution
        """
        df = self.df[self.df[feature].notnull()]
        df_control = list(df[df['genotype'] == 0][feature])
        df_treatment = list(df[df['genotype'] == 1][feature])

        print(feature, 'length control group:', len(df_control), '  |  length treatment:', len(df_treatment))
        p_value = permutation_test(df_treatment, df_control,
                                   method='approximate',
                                   num_rounds=self.nsim,
                                   seed=0)
        return p_value





def plot_topfeatures(models, condition = None, ntop=5):
    bestfeatures = [model.featureImportances['Features'][:ntop] for model in models]  # get tp 5 features
    bestfeatures = [item for sublist in bestfeatures for item in sublist]  # flatten list
    bestfeatures = list(dict.fromkeys(bestfeatures))  # delete duplicates
    bestfeatures.sort()
    _ = StatsPlot(features=bestfeatures, condition=condition)






# df = pd.read_csv('/home/iglohut/Github/PredictModels/Data/SS_alldata_OS_ehmt1.csv')
#
#
# features =['first_object', 'first_object_latency', 'stay1', 'stay2','SS1', 'perseverance']
# features =['min1_DI', 'min2_DI', 'min3_DI', 'min4_DI', 'min5_DI']
# features = ['min1_n_explore','min2_n_explore', 'min3_n_explore', 'min4_n_explore', 'min5_n_explore']
# features = ['min1_obj1_time', 'min2_obj1_time', 'min3_obj1_time', 'min4_obj1_time','min5_obj1_time']
# features = ['n_transitions', 'min2_obj2_time', 'min5_DI']
# _ = StatsPlot(features)

