import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split, Subset

from .dataset import BindingDataset

def train_val_test_split(
        csv_path,
        seq_col,
        score_col,
        tok_model,
        val_split,
        test_split,
        batch_size,
        seed
):
    """
    Loads data from CSV, splits it into training pool, validation set, and test set.

    Returns:
        tuple[Subset, DataLoader, DataLoader]: A tuple containing:
            - training_pool (Subset): The subset of data for active learning.
            - val_dataloader (DataLoader): DataLoader for the validation set.
            - test_dataloader (DataLoader): DataLoader for the test set.
    """
    torch.manual_seed(seed)

    # Load data from CSV
    df = pd.read_csv(csv_path)
    sequences = df[seq_col].tolist()
    scores = df[score_col].tolist()

    # Create full dataset
    full_dataset = BindingDataset(sequences, scores, tok_model)

    # Calculate split sizes
    total_len = len(full_dataset)
    val_len = int(total_len * val_split)
    test_len = int(total_len * test_split)
    train_len = total_len - val_len - test_len

    # Split into training pool, validation, and test sets
    training_pool, val_dataset, test_dataset = random_split(full_dataset, [train_len, val_len, test_len])

    # Create fixed DataLoaders for validation and test sets
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return training_pool, val_dataloader, test_dataloader