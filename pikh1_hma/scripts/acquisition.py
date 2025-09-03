import torch
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path

from .training import initialize_and_train_new_model


def get_pool_predictions(model, pool_dataloader):
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    model.eval()
    all_preds = []

    with torch.inference_mode():
        # iterate through pool loader
        for inputs, labels in tqdm(pool_dataloader, desc=f"[Surveying]"):
            # get model predictions, append them to list (num batches, batch size)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=labels)
            preds = outputs.logits
            all_preds.append(preds.cpu())
    
    # concat predictions from all batches for a single prediction tensor
    all_preds = torch.cat(all_preds)
    return all_preds

# get acquisition scores (variance) given model predictions
def get_variances(ensemble_predictions, save_path=None):
    # calculate variance for each index
    variances = torch.var(ensemble_predictions, dim=0)
    variances = variances.squeeze()

    if save_path:
        print(f"Saving variance distribution to {save_path}...")

        save_path = Path(save_path)
        save_dir = save_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)

        # Move tensor to CPU and convert to a NumPy array to use with pandas
        variances_np = variances.cpu().numpy()
        
        # Create a pandas DataFrame with a descriptive column name
        df = pd.DataFrame(variances_np, columns=['variance'])
        
        # Save the DataFrame to a CSV file, without writing the DataFrame index
        df.to_csv(save_path, index=False)
        print("Save complete.")
    # return list of acquisition scores
    return variances

# acquire new batch, randomly if no scores given, top "batch_size_to_acquire" if given
def acquire_new_batch(dataset, train_dataloader_batch_size, pool_dataloader_batch_size, initial_batch_size, batch_size_to_acquire, labeled_indices, unlabeled_indices, acquisition_scores=None):
    # if initial batch, when there are no acquisition scores, select randomly
    if acquisition_scores is None:
        initial_batch_size = min(initial_batch_size, len(unlabeled_indices))
        indices_to_acquire = np.random.choice(unlabeled_indices, size=initial_batch_size, replace=False)
    
    # else select based on top acquisition scores
    else:
        # make sure we don't overshoot samples to acquire if on the final batch
        batch_size_to_acquire = min(batch_size_to_acquire, len(acquisition_scores))
        # get the indicies of the top acquisition scores (num of samples)
        top_k_indices = acquisition_scores.topk(batch_size_to_acquire).indices
        # use these to find the indicies that map back to the original dataset
        indices_to_acquire = unlabeled_indices[top_k_indices.cpu().numpy()]
    
    # update the indices lists
    labeled_indices = np.concatenate([labeled_indices, indices_to_acquire])
    unlabeled_indices = np.setdiff1d(unlabeled_indices, indices_to_acquire, assume_unique=True)
    
    # create new subsets and dataloaders
    train_subset = Subset(dataset, labeled_indices.tolist())
    pool_subset = Subset(dataset, unlabeled_indices.tolist())
    train_dataloader = DataLoader(train_subset, batch_size=train_dataloader_batch_size, shuffle=True)
    pool_dataloader = DataLoader(pool_subset, batch_size=pool_dataloader_batch_size, shuffle=False)
    
    return train_dataloader, pool_dataloader, labeled_indices, unlabeled_indices

def get_bootstrap_sample(labeled_indices, pool_dataset, train_dataloader_batch_size):
    bootstrap_indices = np.random.choice(labeled_indices, size=int(0.9*len(labeled_indices)),replace=True)
    bootstrap_subset = Subset(pool_dataset, bootstrap_indices)
    bootstrap_dataloader = DataLoader(bootstrap_subset, batch_size=train_dataloader_batch_size, shuffle=True)
    return bootstrap_dataloader

def train_bootstrapped_ensemble(
        n_models, 
        model_name, 
        approach,
        learning_rate,
        weight_decay,
        epochs,
        labeled_indices,
        train_dataloader_batch_size,
        pool_dataset, 
        pool_dataloader, 
        val_dataloader,
        patience
        ):
    
    # define list to store predictions as each model is trained then evaluated
    ensemble_predictions = []
    
    for i in range(n_models):
        print(f"\nTraining Model {i+1}...")
        # set a changing manual seed
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)

        # get bootstrap sample from labeled dataset
        bootstrap_dataloader = get_bootstrap_sample(labeled_indices, pool_dataset, train_dataloader_batch_size)

        # initialize and train a new model
        model = initialize_and_train_new_model(approach, model_name, learning_rate, weight_decay, epochs, bootstrap_dataloader, val_dataloader, patience)
        
        # get model predictions on pool dataloader, append to ensemble predictions list
        pool_preds = get_pool_predictions(model, pool_dataloader, )
        ensemble_predictions.append(pool_preds)

    # stack ensemble predictions to create tensor of shape (n_models, n_unlabeled_samples)
    ensemble_predictions = torch.stack(ensemble_predictions, dim=0)
    print("Ensemble training complete, submitting predictions for next cycle.")
    # return list of ensemble predictions
    return ensemble_predictions