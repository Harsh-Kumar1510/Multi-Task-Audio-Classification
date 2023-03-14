import torch

# Model configuration
model_cfg = {
    'learning_rate' : 0.0001,
    'batch_size' : 16,
    'num_folds': 6,
    'num_epochs':200,
    'pos_weight': 0.3,
    'gender_weight': 0.5,
    'digit_weight': 0.5,
    'log_path':'Logs',  # save val and train loss
    'best_model_path':'BestModelWeight', # location to save best model from each folds
    'device' :'cuda' if torch.cuda.is_available() else 'cpu'
 }
        



