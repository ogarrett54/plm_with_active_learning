import torch
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd


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

# TODO: Add acquisition functions, these will change depending on the approach