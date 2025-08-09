import torch
from torch.nn import MSELoss
from torch.optim import AdamW
from tqdm import tqdm
from torchmetrics.regression import SpearmanCorrCoef, PearsonCorrCoef, MeanSquaredError
from transformers import logging

from .models import get_model

logging.set_verbosity_error()

def train_step(model, optimizer, train_dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    model.train()
    total_train_loss = 0
    for inputs, labels in train_dataloader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_dataloader)
    return avg_train_loss

def val_step(model, val_dataloader, spearman):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.eval()
    total_val_loss = 0

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for inputs, labels in val_dataloader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs, labels=labels)
            preds = outputs.logits.squeeze() # to make sure dimensions are the same for spearman
            loss = outputs.loss

            total_val_loss += loss.item()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    avg_val_loss = total_val_loss / len(val_dataloader)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    spearmanr = spearman(all_preds, all_labels).item()

    return avg_val_loss, spearmanr

def initialize_and_train_new_model(
        approach,
        model_name, 
        learning_rate, 
        weight_decay,
        epochs, 
        train_dataloader, 
        val_dataloader, 
        patience=5,
        return_history=False,
        checkpoint_path="best_model.pth"
        ):
    # TODO: Support new models by adding logic to handle a user-defined or HF-define loss function
    # initialize correct model based on approach defined by user
    model = get_model(approach, model_name)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    spearman = SpearmanCorrCoef()

    # initialize variables for early stopping
    best_val_spearman = -1
    epochs_wo_improvement = 0

    # initialize lists to store metrics
    train_loss_history = []
    val_loss_history = []
    spearmanr_history = []

    # main training loop
    for epoch in tqdm(range(epochs), desc="[Training]"):
        train_loss = train_step(model, optimizer, train_dataloader)
        val_loss, spearmanr = val_step(model, val_dataloader, spearman)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        spearmanr_history.append(spearmanr)

        # early stopping logic
        if spearmanr > best_val_spearman:
            best_val_spearman = spearmanr
            epochs_wo_improvement = 0
            # save the best model for later
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_wo_improvement += 1
        
        if epochs_wo_improvement == patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break
        
    print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | SpearmanR: {spearmanr:.4f}')

    # load the best model before output
    model.load_state_dict(torch.load(checkpoint_path))

    if return_history:
        history = {
            'train_loss': train_loss_history,
            'val_loss': val_loss_history,
            'spearmanr': spearmanr_history
        }
        return model, history

    return model

def test_model(model, test_dataloader, return_results=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize metrics
    spearman = SpearmanCorrCoef().to(device)
    pearson = PearsonCorrCoef().to(device)
    mse = MeanSquaredError().to(device)

    model.to(device)
    model.eval()
    
    total_test_loss = 0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for inputs, labels in tqdm(test_dataloader, desc="[Testing]"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs, labels=labels)
            preds = outputs.logits.squeeze()
            loss = outputs.loss

            total_test_loss += loss.item()

            all_preds.append(preds)
            all_labels.append(labels)

    # Concatenate all predictions and labels from all batches
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Calculate final metrics
    avg_test_loss = total_test_loss / len(test_dataloader)
    spearmanr = spearman(all_preds, all_labels).item()
    pearsonr = pearson(all_preds, all_labels).item()
    final_mse = mse(all_preds, all_labels).item()

    if return_results:
        results = {
            "avg_test_loss": avg_test_loss,
            "spearmanr": spearmanr,
            "pearsonr": pearsonr,
            "final_mse": final_mse
        }
        return results
    else:
        # Print the report
        print(f"Spearman's Rho: {spearmanr:.4f}")
        print(f"Pearson's Rho: {pearsonr:.4f}")
        print(f"Mean Squared Error (MSE): {final_mse:.4f}")