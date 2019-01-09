import pandas as pd
from joblib import dump, load
from os.path import isfile

# Saver specific for the Titanic task
# The Saver saves the predictions to an submission file
class mySaver(object):
    def save_predictions(self, predictions, file_name):
        df = pd.DataFrame(predictions)
        df.columns = ['Gene', 'PredictedGene']
        df.to_csv(file_name, index=False)

        print ("Predictions written to " , file_name)

    def save_models(self, models, test_accuracies):
        # Load saved model scores
        if isfile('./saved_models/saved_model_scores.pkl'):
            saved_model_scores = load('./saved_models/saved_model_scores.pkl')
        else:
            # Create the file from scratch
            open('./saved_models/saved_model_scores.pkl', 'w+').close()
            saved_model_scores = {}

        # Update saved model scores
        for i, model in enumerate(models):
            if model.name not in saved_model_scores or not isfile('./saved_models/' + model.name + '.pkl'):
                print("Model saved to saved_models/" + model.name + ".pkl")
                dump(model, 'saved_models/' + model.name + '.pkl')
                saved_model_scores[model.name] = test_accuracies[i]
            elif test_accuracies[i] > saved_model_scores[model.name]:
                print("Model saved to saved_models/" + model.name + ".pkl")
                dump(model, 'saved_models/' + model.name + '.pkl')
                saved_model_scores[model.name] = test_accuracies[i]

        # Save model scores to disk
        print(saved_model_scores)
        dump(saved_model_scores, './saved_models/saved_model_scores.pkl')