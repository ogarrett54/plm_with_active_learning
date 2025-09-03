import torch
from tqdm import tqdm
import pandas as pd
from pathlib import Path

def get_residuals(model, pool_dataloader, save_path=None):
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    model.eval()
    all_residuals = []

    with torch.inference_mode():
        # iterate through pool loader
        for inputs, labels in tqdm(pool_dataloader, desc=f"[Getting Residuals]"):
            # get model predictions, append them to list (num batches, batch size)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            outputs = model(**inputs, labels=labels)
            preds = outputs.logits.squeeze()
            residuals = abs(labels - preds)
            all_residuals.append(residuals.cpu())

    # concat predictions from all batches for a single prediction tensor
    all_residuals = torch.cat(all_residuals)

    if save_path:
        print(f"Saving residuals to {save_path}...")

        save_path = Path(save_path)
        save_dir = save_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Move tensor to CPU and convert to a NumPy array to use with pandas
        residuals_np = all_residuals.cpu().numpy()
        
        # Create a pandas DataFrame with a descriptive column name
        df = pd.DataFrame(residuals_np, columns=['residuals'])
        
        # Save the DataFrame to a CSV file, without writing the DataFrame index
        df.to_csv(save_path, index=False)

    return all_residuals

def get_chosen_labels_and_preds(model, train_dataloader, save_path=None):
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    model.eval()
    all_labels = []
    all_preds = []

    with torch.inference_mode():
        # iterate through pool loader
        for inputs, labels in tqdm(train_dataloader, desc=f"[Getting Chosen Labels and Preds]"):
            # get model predictions, append them to list (num batches, batch size)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=labels)
            preds = outputs.logits
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # concat predictions from all batches for a single prediction tensor
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    if save_path:
        print(f"Saving residuals to {save_path}...")

        save_path = Path(save_path)
        save_dir = save_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)

        # Move tensor to CPU and convert to a NumPy array to use with pandas
        labels_np = all_labels.cpu().numpy()
        preds_np = all_preds.cpu().numpy()
        
        # Create a pandas DataFrame with a descriptive column name
        df = pd.DataFrame(columns=['labels', 'preds'])
        df.labels = labels_np
        df.preds = preds_np

        # Save the DataFrame to a CSV file, without writing the DataFrame index
        df.to_csv(save_path, index=False)
    return all_labels, all_preds