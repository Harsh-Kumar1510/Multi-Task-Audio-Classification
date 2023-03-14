import pickle 

from utils import confusionMatrixPlot

def savePlot(prediction:str, location:str) -> None:
    """
    Save the confusion matrix plots under specific location

    Args:
       prediction (str) : Path for prediction pikle  file.
       location(str): Location save the confusion matrix plot.
    """

    with open(prediction, 'rb') as f:
        data = pickle.load(f)
        # print(data)
        
        for  fold in data:
            for fold_name, fold_predict in fold.items():

                # Save confusion matrix plot for digit classification
                confusionMatrixPlot(trueLabel=fold_predict['digit_gt'],
                                    predLabel=fold_predict['digit_predict'],
                                    location= location + f'Digit_{fold_name}')
                

                # Save cofusion matrix plot for gender classification
                confusionMatrixPlot(trueLabel=fold_predict['gen_gt'],
                                    predLabel=fold_predict['gen_predict'],
                                    location= location+ f'Gender_{fold_name}')
                



if __name__ == '__main__':
    # Location to save the plot
    location = 'Result/ConfusionMatrix/'

    # Prediction pickle file
    prediction = 'Model/BestModelWeight/cv_prediction.pickle'

    # Save the plot
    savePlot(prediction, location)