from joblib import load

# Loader specific for the Titanic task
# The loader loads the data
class myLoader(object):
    def load_pkl(self, file_name):
        return load(file_name)