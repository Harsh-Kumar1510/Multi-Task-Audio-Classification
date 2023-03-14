import numpy as np
import pickle
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from typing import List, Tuple, Dict, Any, Union



def cvFolderSplits(mainPath:str, k:int=0, num_folds:int=6) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Splits all male and female folders into k folds for cross-validation in such a way that both male and
    female have same porportion in all data splits.
    
    Args:
        mainPath (str): Path to the preprocessed parent directory containing the subject folders.
        k (int): Index of the fold to use as the test set (range: 0 to num_folds-1).
        num_folds (int): Number of folds to use for cross-validation.

    Returns:
        (train_dirs, val_dirs, test_dirs): A tuple containing the train, validation, and test directories.
    """
    # Check k
    if k >= num_folds:
        raise ValueError(f'k must be less than {num_folds}')


    # Main folder path
    parent_dir = Path(mainPath) 

    # Get female and male directories
    female_dir = ['12_pickled','26_pickled','28_pickled','36_pickled','43_pickled','47_pickled','52_pickled','56_pickled','57_pickled','58_pickled','59_pickled','60_pickled'] # list of female directories
    female_dir_list = [dir for dir in parent_dir.iterdir() if dir.is_dir() and dir.parts[-1] in female_dir] # Female directory full paths
    male_dir_list = [dir for dir in parent_dir.iterdir() if dir.is_dir() and dir.parts[-1] not in female_dir] # Male directory full paths

    # Shuffle male and female directory separaretly using a fixed random seed before CV splits
    np.random.seed(42)
    np.random.shuffle(male_dir_list)
    np.random.shuffle(female_dir_list)


    # Calculate total male and female folders in each fold
    male_fold_size = len(male_dir_list) // num_folds
    female_fold_size = len(female_dir_list) // num_folds

    # Male directories
    start_idx_m = k * male_fold_size
    end_idx_m = (k + 1) * male_fold_size
    test_male_fold = male_dir_list[start_idx_m:end_idx_m]
    test_male_fold = male_dir_list[start_idx_m:end_idx_m] # test set

    # Val and train set at last fold
    if k == num_folds-1:
        val_male_fold = male_dir_list[:male_fold_size] # take first folds as val set
        train_male_fold = np.concatenate([male_dir_list[male_fold_size:start_idx_m], male_dir_list[end_idx_m+male_fold_size:]]) # train set
    # Otherwise
    else:
        val_male_fold = male_dir_list[start_idx_m + male_fold_size : end_idx_m + male_fold_size] # val set
        train_male_fold = np.concatenate([male_dir_list[:start_idx_m], male_dir_list[end_idx_m+male_fold_size:]]) # train set
    

    # Female directories
    start_idx_f = k * female_fold_size # start index
    end_idx_f = (k+1) * female_fold_size # end index
    test_female_fold = female_dir_list[start_idx_f:end_idx_f] # test set
    
    # Val and train set at last fold
    if k == num_folds-1:
        val_female_fold = female_dir_list[:female_fold_size] # take first folds as val set
        train_female_fold = np.concatenate([female_dir_list[female_fold_size:start_idx_f], female_dir_list[end_idx_f+female_fold_size:]]) # train set
    # Otherwise 
    else:
        val_female_fold = female_dir_list[start_idx_f + female_fold_size :end_idx_f+female_fold_size]
        train_female_fold = np.concatenate([female_dir_list[:start_idx_f], female_dir_list[end_idx_f+female_fold_size:]]) # train set


    # Combine both male and female directories 
    test_dirs = np.concatenate([test_male_fold, test_female_fold]) # Test data
    val_dirs = np.concatenate([val_male_fold, val_female_fold]) # Val data
    train_dirs = np.concatenate([train_male_fold, train_female_fold]) # Train dat

    return train_dirs, val_dirs, test_dirs


def savePrediction(data: List[Dict[str, Any]], filePath: str) -> None:
    """
    Saves prediction using the pickle module.

    Args:
        data (List[Dict[str, Any]]): A list of prediction from each folds.
        filename (str): File location to save.
    """
    with open(filePath, 'wb') as f:
        pickle.dump(data, f)


def computeSTD(data: Union[List, np.ndarray]) -> Tuple[float, float]:
    """
    Computes the mean and standard deviation of the given data.

    Args:
        data (Union[List, np.ndarray]): Data to compute std.

    Returns:
        float: std of  given data.
    """
    if isinstance(data, np.ndarray):
        mean = np.round(data.mean(), 3)
        std = np.round(data.std(), 3)
        return mean, std
    
    elif isinstance(data, list):
        data_numpy= np.array(data)
        mean = np.round(data_numpy.mean(), 3)
        std = np.round(data_numpy.std(), 3)
        return mean, std
    
    else:
        raise TypeError("Input data must be either a list or numpy ndarray.")
    



def confusionMatrixPlot(trueLabel:Union[list,np.ndarray], 
                        predLabel:Union[list,np.ndarray],
                        location: str,
                        figsize:Tuple[float, float]=None,
                        labels:List[Union[str, int]]=None)->None:
    """ 
    Visualize the confusion matrix for the given data.
    Args:
        trueLabel (Union[list,np.ndarray]): True labels of the data.
        predLabel (Union[list,np.ndarray]): Predicted labels of the data.
        location (str): Title of the plot.
        figsize (Tuple[float, float], optional): Size of the figure. Default is (7,5).
        labels (List[Union[str, int]], optional): List of labels for each class.
        If not provided, numeric labels are used.
    """
    
    # Compute confusion matrix
    cf = confusion_matrix(y_true=trueLabel, y_pred=predLabel,normalize=None)

    # Set default figure size if not provided
    if figsize is None:
        figsize = (7,5)

    # Set default labels if label not provided
    if labels is None:
        labels = [str(i) for i in range(cf.shape[0])] 


    # Vizualize
    plt.figure(figsize= figsize)
    sns.heatmap(cf, annot=True, xticklabels=labels, yticklabels=labels, fmt='g', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.savefig(location)
    plt.show()


   