import pandas as pd
import numpy as np
from scipy import stats


class ImportData():
    file = './Data/SS_alldata_OS_ehmt1.csv'
    features = ["first_object", "first_object_latency", "stay1", "stay2", "SS1", "perseverance", "n_transitions",
                "min1_n_explore", "min2_n_explore", "min3_n_explore", "min4_n_explore", "min5_n_explore",
                'min1_n_explore_obj1', 'min2_n_explore_obj1', 'min3_n_explore_obj1', 'min4_n_explore_obj1',
                'min5_n_explore_obj1',
                'min1_n_explore_obj2', 'min2_n_explore_obj2', 'min3_n_explore_obj2', 'min4_n_explore_obj2', 'min5_n_explore_obj2',
                "min1_obj1_time", "min2_obj1_time", "min3_obj1_time", "min4_obj1_time", "min5_obj1_time",
                "min1_obj2_time", "min2_obj2_time", "min3_obj2_time", "min4_obj2_time", "min5_obj2_time",
                "min1_DI", "min2_DI", "min3_DI", "min4_DI", "min5_DI",
                "min1_explore_time", "min2_explore_time", "min3_explore_time", "min4_explore_time", "min5_explore_time",
                "bout_time", "bout_obj1_time", "bout_obj2_time"]

    # Try fewer featues?
    # features = ["first_object", "first_object_latency", "stay1", "stay2", "SS1", "perseverance", "n_transitions",
    #             "min1_n_explore", "min2_n_explore", "min3_n_explore", "min4_n_explore", "min5_n_explore",
    #             'min1_n_explore_obj1', 'min2_n_explore_obj1', 'min3_n_explore_obj1', 'min4_n_explore_obj1',
    #             'min5_n_explore_obj1',
    #             'min1_n_explore_obj2', 'min2_n_explore_obj2', 'min3_n_explore_obj2', 'min4_n_explore_obj2', 'min5_n_explore_obj2',
    #             "min1_DI", "min2_DI", "min3_DI", "min4_DI", "min5_DI",
    #             "min5_explore_time",
    #             "bout_time", "bout_obj1_time", "bout_obj2_time"]

    target = ["genotype"]


    def __init__(self, condition = None, remove_outliers=None):
        self.df = self.load_dataset(remove_outliers=remove_outliers)
        if condition is not None:
            self.df = self.df[self.df['condition'] == condition]

        self.X = self.df[self.features]
        self.y = self.df[self.target]


    def load_dataset(self, remove_outliers = None):
        df = pd.read_csv('./Data/SS_alldata_OS_ehmt1.csv')
        df = df.loc[~df.subject.isin([8, 9, 10, 11, 12] + list(range(100, 120)))]  # Don't go over round 2/8 subjects
        fillFeatures = ['stay1', 'stay2', 'SS1']  # Features that could be nan if mouse never switched object
        df[fillFeatures] = df[fillFeatures].fillna(-1)  # Fill the nans with -1: choose more logical value?
        df['condition'].replace(['con', 'od', 'or'], [0, 1, 2], inplace=True)

        df = df[self.features + self.target + ['condition']]
        df = df.apply(lambda x: [y if y < np.exp(10) else np.nan for y in x])  # calculatione errors that result in near infinite values
        df = df.dropna()

        if remove_outliers is not None:
            df = df[(np.abs(stats.zscore(df[self.features + self.target])) < 3).all(axis=1)]  # This removes outliers!

        df['condition'].replace([0, 1, 2], ['con', 'od', 'or'], inplace=True)
        return df