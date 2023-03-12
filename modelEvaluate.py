import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda, no_grad
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import  Dict



def evaluate(model:nn.Module, test_data:DataLoader, device:str)->Dict:
    """
    Evaluate the performance of the model on the test dataset.

    Args:
        model (nn.Module): The PyTorch model to be evaluated.
        test_data (DataLoader): The PyTorch DataLoader containing the test data.
        device (str): The device to be used for the computation (e.g., 'cpu' or 'cuda').

    Returns:
        Dict: A dictionary containing the following key-value pairs:
            - 'accuracy_digit': Accuracy of the digit classification task as a percentage.
            - 'accuracy_gender': Accuracy of the gender classification task as a percentage.
            - 'digit_predict': A numpy array containing the predicted digit labels.
            - 'digit_gt': A numpy array containing the ground truth digit labels.
            - 'gen_predict': A numpy array containing the predicted gender labels.
            - 'gen_gt': A numpy array containing the ground truth gender labels.
    """

    # Set model to evaluation mode
    model.eval()

    # Keep tract correct prediction
    correct_digit = 0
    correct_gen = 0
    total = 0

    # Collect prediction for confusion matrix
    digit_predict = []
    digit_gt =[]
    gen_predict = []
    gen_gt = []

    # Disable gradient calculation 
    with torch.no_grad():
        for batch in test_data:
            # Get features and labels
            test_feat = batch['feature'].float()
            digit_label = batch['digit_label']
            digit_label = torch.argmax(digit_label, dim=1).long() # label indices 
            gen_label = batch['gen_label'].float()

            # Give them to the appropriate device
            test_feat = test_feat.to(device)
            digit_label = digit_label.to(device)
            gen_label = gen_label.to(device)
            
            # Make the prediction
            gen_pred, digit_pred = model(test_feat)
            
            # Gender prediction: above 0.5 th male, otherwise female class
            gen_pred = (F.sigmoid(gen_pred) > 0.5).int().squeeze(dim=1) 
            gen_label = gen_label.int().squeeze(dim=1)
            gen_predict.extend(gen_pred.cpu().numpy())
            gen_gt.extend(gen_label.cpu().numpy())
            
            # Digit prediction class indices
            digit_pred = digit_pred.argmax(dim=1) 
            digit_predict.extend(digit_pred.cpu().numpy())
            digit_gt.extend(digit_label.cpu().numpy())
            
            # Keep track  correct prediction
            total += digit_label.shape[0] # total samples
            correct_digit += (digit_pred == digit_label).sum().item() # digit
            correct_gen += (gen_pred == gen_label).sum().item() # gender
    
    accuracy_digit = 100 * correct_digit / total
    accuracy_gen = 100 * correct_gen / total
    print(f'Accuracy on digit classificaiton: {accuracy_digit: .3f}')
    print(f'Accuracy on gender classificaiton: {accuracy_gen: .3f}')

    return {"accuracy_digit": accuracy_digit,
            "accuracy_gender": accuracy_gen,
            "digit_predict": np.array(digit_predict), 
            "digit_gt": np.array(digit_gt), 
            "gen_predict": np.array(gen_predict),
            "gen_gt": np.array(gen_gt)}
    




