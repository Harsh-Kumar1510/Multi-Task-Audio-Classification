import numpy as np
from pathlib import Path
import torch
from torch import cuda
from torch.utils.data import DataLoader

from pytorch_dataset import MyMultitaskDataset
from Model.model_training import trainModel
from Model.model import MultitaskCNN
from Model.model_evaluate import evaluate
from Model.config import data_path, model_cfg
from utils import *




def main(mainPath:str, device:str,  modelPath:str, logPath:str, numFolds:int=6):
    """
    Trains, evaluates, and stores the best model for each fold of a k-fold cross-validation.

    Args:
        mainPath (str): The path to the parent directory containing the subject folders.
        device (str): The device on which to run the model (e.g., 'cpu' or 'cuda').
        modelPath (str): The location to save the best model from each fold.
        logPath (str): The directory to save the validation and training logs from each epoch.
        numFolds (int): The number of folds to use for cross-validation. Default: 6.
    """
   
    # Store  accuracy from each CV
    cv_accuracy_digit = []
    cv_accuracy_gender = []

    # Store predicton from each CV (this is for confusion matrix)
    prediction_cv = []

    # Start training under Cross-validation (CV)
    for k in range(numFolds):
        print(f'Fold-{k+1} is processing....')

        # Get each CV splits train, val and test folder paths
        train_dirs, val_dirs, test_dirs = cvFolderSplits(mainPath=mainPath, 
                                                         k=k, 
                                                         num_folds=numFolds)

        
        # Custom Dataset and Data loader for Training
        train_dataset = MyMultitaskDataset(train_dirs)
        print(f'Total train samples: {train_dataset.__len__()}')
        trainDataloader = DataLoader(train_dataset, batch_size=16, shuffle=True,drop_last=True)

        # Custom Dataset and Data loader for Validation
        val_dataset = MyMultitaskDataset(val_dirs)
        print(f'Total val samples: {val_dataset.__len__()}')
        valDataloader = DataLoader(val_dataset,batch_size=16, shuffle=True, drop_last=True)

        # Custom Dataset and Data loader for Testing
        test_dataset = MyMultitaskDataset(test_dirs)
        print(f'Total test samples: {test_dataset.__len__()}')
        testDataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, drop_last=True)
        print()


        # Start Training model
        trainModel(trainData=trainDataloader,
                   valData=valDataloader, 
                   device=device, 
                   bestModelPath=modelPath,
                   logPath=logPath,
                   fold=k+1)
        

        # Evaluate on test data
        model = MultitaskCNN() # Initiate Model
        model.load_state_dict(torch.load(f'{modelPath}/best_model_{k+1}.pth')) # Load the best model
        model.to(device) # Specify device
        predict = evaluate(model,testDataloader, device) # Predict
        prediction_cv.append({'fold_'+ str(k+1):predict}) # store prediction
        cv_accuracy_digit.append(predict['accuracy_digit']) # Get digit accuracy and store
        cv_accuracy_gender.append(predict['accuracy_gender']) # Get gender accuracy and store


    # Save Prediction results from all folds
    savePrediction(prediction_cv, modelPath)

    # Print average accuracy from CV
    print(f'Avearge cv_accuracy_digit:  {np.array(cv_accuracy_digit).mean(): .3f}')
    print(f'Avearge cv_accuracy_gender:  {np.array(cv_accuracy_gender).mean(): .3f}')


if __name__ == "__main__":

    # Check if CUDA is available, else use CPU
    device = model_cfg['device']
    print(f'Process on {device}', end='\n\n')
    
    # This need to replace as per your folder structures
    mainPath = 'parent directory that contains all the folders'
    modelPath = data_path['best_model_path'] # location to save best models and results
    logPath = 'directory name for train and val loss to save'
    numFolds = model_cfg['num_folds'] # fold size
    main(mainPath=mainPath, device=device, modelPath=modelPath,logPath=logPath,numFolds=numFolds)

