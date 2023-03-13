from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

from Model.model import MultitaskCNN
from Model.config import model_cfg



def predictSample(model: nn.Module, audio: torch.Tensor) -> Tuple[int, int]:
    """
    Predicts the gender and digit of a given audio sample.

    Args:
        model (nn.Module): The trained PyTorch model.
        audio (torch.Tensor): The audio sample (mel spectogram) to predict.

    Returns:
        A tuple containing the predicted gender (0 or 1) and digit (0 to 9).
    """
    # Set Evaluation mode
    model.eval()

    # Disable autograd calculations and make prediction
    with torch.no_grad():
        gen_pred, digit_pred = model(audio)

    # Gender prediction: above 0.5 th male, otherwise female class
    gen_pred = (torch.sigmoid(gen_pred) > 0.5).int().item()

    # Digit prediction class indices with highest logit
    digit_pred = digit_pred.argmax(dim=1).item()

    return gen_pred, digit_pred


if __name__ == '__main__':

    # Initialize model
    model = MultitaskCNN() 
    device = model_cfg['device']  # Specify device
    model.to(device)  # Move model to the specified device

    model_path = Path('Model/BestModelWeight/best_model_1.pth')  # Model path
    model_weights = model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))  # Load model weight

    # Create dummy audio data (Replace with actual audio mel spectrogram)
    audio_feature = torch.rand(1, 60, 40)

    # Predict 
    gen, digit = predictSample(model, audio_feature.to(device))
    print(f'Gender: {gen}, Spoken Digit: {digit}')