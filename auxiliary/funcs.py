from collections import Iterable
import numpy as np
def flatten(lst):
    def flat(lst):
        for parent in lst:
             if not isinstance(parent, Iterable):
                yield parent
             else:
                 for child in flat(parent):
                    yield child
    return list(flat(lst))

def normalize(x):
    "Normalized al value sbetween 0 and 1."
    x = np.array(x)
    x_normalized = np.array((x-min(x))/(max(x)-min(x)))
    return x_normalized

def sum_1(x):
    "Scales all values to sum up to 1."
    x = np.array(x)
    scaled_x = (x - x.min()) / (x - x.min()).sum()
    return scaled_x