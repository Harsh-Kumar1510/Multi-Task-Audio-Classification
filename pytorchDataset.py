import numpy as np
import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Union


class MyMultitaskDataset(Dataset):
    """
    Custom Dataset class that loads pickle files from the given folder path and reads features, digit label, and gender label from the files. The features are zero-padded or truncated to a fixed length of 60 time-frames and are then normalized to [-1, 1]. 

    Args:
    folder_path (List[Path]): List of paths to folders containing pickle files.
    load_into_memory (bool): If True, the entire dataset is loaded into memory. Otherwise, the dataset is read 
    on-the-fly from disk.
    """

    def __init__(self, folder_path:List[Path], load_into_memory: bool = True)-> None:
        self.folder = folder_path
        self.load_into_memory = load_into_memory
        self.data = []
        # self.max_len = 0

        # Go all pickel file in  each subject folder
        for folder in self.folder:
          pickle_files = list(folder.glob('*.pickle'))
          
          # Go over all files
          for file in pickle_files:

            # Skip if file is empty
            if file.stat().st_size == 0:
                continue

            # Load whole data into memory
            if self.load_into_memory:
              with open(file, 'rb') as f:
                data = pickle.load(f)
                self.data.append(data)

                # Check length of featatures
                # if   data['features'].shape[1] > self.max_len:
                #   self.max_len = data['features'].shape[1]
              
            # Otherwise save only file paths
            else:
              self.data.append(file)
          
       
    def __len__(self)-> int:
        """
        Returns the length of the dataset.
        """
        return len(self.data)


    def __getitem__(self, index)-> Dict[str, Union[np.ndarray, int]]:
        """
        Returns a sample from the dataset at the given index.

        Args:
        index (int): index of the sample.

        Returns:
        Dict[str, Union[np.ndarray, int]]: A dictionary containing the following key-value pairs:
                                            - 'feature': A numpy array of shape (60, 40), containing the
                                                         features of the audio file.
                                            - 'digit_label': A tensor of shape (10,), containing the digit label
                                                             of the audio file.
                                            - 'gen_label': A tensor of shape (1,), containing the gender label
                                                           of the audio file.
        """
  
        # Load the data from memory  or from on fly from disk 
        if self.load_into_memory:
          sample = self.data[index]
        else:
          file = self.data[index] 
          with open(file, 'rb') as f:
                sample = pickle.load(f)
          sample = self.data[index]
          

        # Pad zeros to short features, and truncate the long one
        feature = sample['features']
        if feature.shape[1] < 60:
            pad_width = ((0, 0), (0, 60-feature.shape[1])) 
            feature = np.pad(feature, pad_width, mode='constant', constant_values=0) 
        else:
            feature = feature[:,:60]
        
        # Scale data -1 to 1
        feature = feature * 1/np.max(np.abs(feature))

        # Change from (freq x time-frame)  to (time-fram x freq)
        feature = feature.T

        # Get digit label 
        digit_label = torch.from_numpy(sample['class'][:10])
       
        # Make gender label such that male =1 and female=0
        gen_label = torch.tensor([1 if sample['class'][10:][0] == 1 else 0], dtype=torch.float32)
      
        return {'feature':feature, 'digit_label':digit_label, 'gen_label':gen_label}




