import torch
from torch.utils.data import Subset, DataLoader

from training import initialize_and_train_new_model

def run_standard_finetuning(
        n_samples, 
        model_name,
        batch_size, 
        learning_rate, 
        weight_decay, 
        epochs, 
        training_pool, 
        val_dataloader,
        patience,
        device
        ):
    # get dataloader of random train data
    random_indices = torch.randperm(len(training_pool))[:n_samples].tolist()
    train_subset = Subset(training_pool, random_indices)
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    # train model
    model, history = initialize_and_train_new_model(model_name, learning_rate, weight_decay, epochs, train_dataloader, val_dataloader, patience, return_history=True)
    return model, history


# TODO: Add other campaign runners, these will change depending on the approach