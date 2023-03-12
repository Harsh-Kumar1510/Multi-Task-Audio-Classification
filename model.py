
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda, no_grad
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict, Union



class MultitaskCNN(nn.Module):
    """ A CNN based Muli-task model for gender classifcation and spoken digit classification. """

    def __init__(self):
        super(MultitaskCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)) # Non-overlapping pooling
        self.dropout1 = nn.Dropout(p=0.2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.dropout2 = nn.Dropout(p=0.3)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.dropout3 = nn.Dropout(p=0.4)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.dropout4 = nn.Dropout(p=0.5)

        
        # Feedforward Network
        self.fc1 = nn.Linear(in_features= 256*1, out_features= 128)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=0.5)
        
        # Gender classifier layer
        self.genderClassifer = nn.Linear(in_features= 128, out_features= 1)

        # Digit classifier layer
        self.digitClassifer = nn.Linear(in_features= 128, out_features= 10)

        
    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass of the network on the given input tensor `x` and returns the logits for
        gender and digit classification as a tuple of tensors.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, channel, freq_bins, time_frames) or 
        (batch_size, time_frames, freq_bins). In the latter case, an additional channel dimension is added.

        Returns:
        - A tuple of two tensors representing the logits for gender and digit classification, respectively.
        - The gender logits tensor has shape (batch_size, 1) and digit logits tensor has shape (batch_size, 10).
        """
        
        # Ensure input data is 4D; otherwise add an extra dimension at position 1
        x = x if x.ndimension() == 4 else x.unsqueeze(1)

        # First CNN layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second CNN layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third CNN layer
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # Fourth CNN layer
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = self.dropout4(x)
       
        # Max pooling along the time-frame
        x = torch.max(x, dim=2)[0]
        x = x.squeeze(-1) # (batch, channels)

        # Feedforward layer
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout5(x)
    
        logit_digit = self.digitClassifer(x)
        logit_gender = self.genderClassifer(x)

        return logit_gender, logit_digit




