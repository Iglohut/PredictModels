from collections import Iterable
def flatten(lst):
    def flat(lst):
        for parent in lst:
             if not isinstance(parent, Iterable):
                yield parent
             else:
                 for child in flat(parent):
                    yield child
    return list(flat(lst))
