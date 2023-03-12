import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda, no_grad
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import MultitaskCNN


def trainModel( trainData: DataLoader, valData:DataLoader, device:str, logPath:str, fold:int=1,)-> None:
    """
    Training Helper function.

    Args:
        trainData (DataLoader): training dataset
        valData (DataLoader): validation dataset
        device (str): device to train the model on
        fold (int): fold number, used for tracking and logging
    """
  
    # Define hyperparameters
    num_epochs = 200
    batch_size = 16
    learning_rate = 0.0001
    gender_weight = 0.5
    digit_weight = 0.5

    # Initialize model and optimizer
    model = MultitaskCNN().to(device)
    optimizer = Adam(params=model.parameters(), lr=learning_rate)

    # Define loss function for each task
    # Assign less than 1 pos-weigth to have better precision
    gender_criterion = nn.BCEWithLogitsLoss(pos_weight= torch.tensor([0.3]).to(device)) 
    digit_criterion = nn.CrossEntropyLoss() 

    # Set up for early stopping 
    best_val_loss = float('inf')
    epochs_since_improvement = 0
    max_epochs_no_improvement = 25
    best_epoch = 0  

    # Create a summary writer object (This is for tensorboard visualization)
    writer = SummaryWriter(log_dir=logPath)
    example_data = torch.rand(16, 1, 60, 40) # create a sample data
    writer.add_graph(model, example_data.to(device))
    

    # Training loop
    for epoch in range(num_epochs):

        # Training mode
        model.train()

        # Training loss
        train_joint_loss = 0.0
        train_gender_loss = 0.0
        train_digit_loss = 0.0
        
        # Validation loss
        val_joint_loss = 0.0
        val_gender_loss = 0.0
        val_digit_loss = 0.0
        
        # For each batch of  dataset
        for batch in trainData:

            # Get features and labels
            train_feat = batch['feature'].float()
            digit_label = batch['digit_label']
            digit_label = torch.argmax(digit_label, dim=1).long() # label indices 
            gen_label = batch['gen_label'].float()

            # Give them to the appropriate device.
            train_feat = train_feat.to(device)
            digit_label = digit_label.to(device)
            gen_label = gen_label.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            gend_pred, digit_pred = model(train_feat)
        
            # Compute training loss for each task
            digit_loss = digit_criterion(digit_pred, digit_label)
            gender_loss = gender_criterion(gend_pred, gen_label)
            
            # Compute the joint loss
            joint_loss = digit_weight * digit_loss + gender_weight* gender_loss
            
            # Backward pass and weights optimization
            joint_loss.backward()
            optimizer.step()

            # Update epoch loss
            train_joint_loss += joint_loss.item()
            train_gender_loss += gender_loss.item()
            train_digit_loss += digit_loss.item()

        # Show average training loss in each epoch
        batch_size = trainData.batch_size
        num_batches = len(trainData)
        total_samples = batch_size * num_batches
        avg_joint_loss = train_joint_loss / total_samples
        avg_gender_loss = train_gender_loss / total_samples
        avg_digit_loss = train_digit_loss / total_samples
        print(f'Epoch: {epoch:03d} | '
            f'Mean joint train loss: {avg_joint_loss:.5f} | '
            f'Mean digit train loss {avg_digit_loss:.5f} | '
            f'Mean gender train loss {avg_gender_loss:.5f}')
            
    
        # Evaluation  mode
        model.eval()

        # Say to PyTorch not to calculate gradients
        with no_grad():

            # For each batch of our dataset
            for batch in valData:
                # Get features and labels
                val_feat = batch['feature'].float()
                digit_label = batch['digit_label']
                digit_label = torch.argmax(digit_label, dim=1).long() # label indices 
                gen_label = batch['gen_label'].float()
            
                # Give them to the appropriate device
                val_feat = val_feat.to(device)
                digit_label = digit_label.to(device)
                gen_label = gen_label.to(device)

                # Make the prediction
                gen_pred, digit_pred = model(val_feat)
                
                # Compute valiation loss for each task
                digit_loss = digit_criterion(digit_pred,digit_label)
                gender_loss = gender_criterion(gen_pred,gen_label)

                # Compute the joint val loss
                joint_loss = digit_weight * digit_loss + gender_weight* gender_loss

                # Update epoch val loss
                val_joint_loss += joint_loss.item()
                val_gender_loss += gender_loss.item()
                val_digit_loss += digit_loss.item()

            # Show average validation loss in each epoch
            batch_size = valData.batch_size
            num_batches = len(valData)
            total_samples = batch_size * num_batches
            avg_val_joint_loss = val_joint_loss / total_samples
            avg_val_gender_loss = val_gender_loss / total_samples
            avg_val_digit_loss = val_digit_loss / total_samples
            
            print(f'Epoch: {epoch:03d} | '
                f'Mean joint valid loss: {avg_val_joint_loss:.5f} | '
                f'Mean digit valid loss {avg_val_digit_loss:.5f}  | ' 
                f'Mean gender valid loss {avg_val_gender_loss:.5f}')
            print()

            # Check early stopping condition
            if avg_val_joint_loss <  best_val_loss:
                best_val_loss = joint_loss
                epochs_since_improvement = 0 # reset
                best_epoch = epoch
                # Save the best model
                torch.save(model.state_dict(), f'{logPath}_{fold}.pth')
            else:
                epochs_since_improvement += 1 # update

            # Check too many epochs without improvement
            if epochs_since_improvement == max_epochs_no_improvement:
                print(f"Training stopped after {epoch} epochs due to no improvement.")
                break

        # Write to tensorboard
        writer.add_scalars(main_tag=f'Loss_{fold}', 
                            tag_scalar_dict={'train': avg_joint_loss, 'val':avg_val_joint_loss},
                            global_step=epoch)
        

    # Close the writer
    writer.close()
    print(f'Best epoch: {best_epoch}')
