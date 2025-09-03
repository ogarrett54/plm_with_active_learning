import torch
from torch.utils.data import Subset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np


from .training import initialize_and_train_new_model
from .acquisition import acquire_new_batch, train_bootstrapped_ensemble, get_variances
from .training import test_model

def run_standard_finetuning(
        n_samples,
        approach, 
        model_name,
        batch_size, 
        learning_rate, 
        weight_decay, 
        epochs, 
        training_pool, 
        val_dataloader,
        patience
        ):

    # get dataloader of random train data
    random_indices = torch.randperm(len(training_pool))[:n_samples].tolist()
    train_subset = Subset(training_pool, random_indices)
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    # train model
    model, history = initialize_and_train_new_model(approach, model_name, learning_rate, weight_decay, epochs, train_dataloader, val_dataloader, patience, return_history=True)
    return model, history

def get_bootstrapped_ensemble_learning_curves(
        n_samples,
        initial_n_samples,
        n_samples_per_batch,
        model_name, 
        approach,
        learning_rate, 
        weight_decay, 
        epochs, 
        training_pool, 
        train_dataloader_batch_size,
        pool_dataloader_batch_size,
        val_dataloader, 
        test_dataloader,
        patience=5,
        n_models=5,
        results_path="results/active_vs_standard_learning_curves.csv"
):
    results_path = Path(results_path)
    results_dir = results_path.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load existing results if the file exists, otherwise start with a fresh DataFrame.
    if results_path.exists():
        all_results_df = pd.read_csv(results_path)
    else:
        all_results_df = pd.DataFrame()
    
    total_pool_size = len(training_pool)
    unlabeled_indices = np.arange(total_pool_size)
    labeled_indices = np.array([], dtype=np.int64)

    ensemble_predictions = None
    current_cycle = 1
    total_cycles = int(np.ceil((n_samples-initial_n_samples)/n_samples_per_batch)) + 1
    
    while len(labeled_indices) < n_samples and len(unlabeled_indices) > 0:
        print(f"\nCycle {current_cycle}/{total_cycles}\n-------------------------------------------------")

        # on the first cycle, choose random samples of initial_n_samples size
        if ensemble_predictions is None:
            print(f"Choosing initial {initial_n_samples} samples randomly...")
            train_dataloader, pool_dataloader, labeled_indices, unlabeled_indices = acquire_new_batch(
                training_pool, train_dataloader_batch_size, pool_dataloader_batch_size, initial_n_samples, n_samples_per_batch, labeled_indices, unlabeled_indices, acquisition_scores=None
            )
        # each other time, use the n_samples_per_batch with acquisition scores to select
        else:
            scores = get_variances(ensemble_predictions, f"{str(results_dir)}/variances{current_cycle}.csv")
            print(f"Selecting new data points...")
            train_dataloader, pool_dataloader, labeled_indices, unlabeled_indices = acquire_new_batch(
                training_pool, train_dataloader_batch_size, pool_dataloader_batch_size, initial_n_samples, n_samples_per_batch, labeled_indices, unlabeled_indices, acquisition_scores=scores
            )
        
        # give message when loop ends
        if len(unlabeled_indices) == 0:
            print("Unlabeled pool is empty. Proceeding to final model training.")
            break
        
        # evaluate active vs standard
        final_results = []

        # active
        print(f"\nTraining and evaluating model using {len(labeled_indices)} actively selected samples...")
        model_active = initialize_and_train_new_model(approach, model_name, learning_rate, weight_decay, epochs, train_dataloader, val_dataloader, patience, return_history=False)
        results_active = test_model(model_active, test_dataloader, return_results=True)
        results_active = {
            'changing_var': 'n_samples',
            'local_exp_idx': current_cycle-1,
            'value': len(labeled_indices),
            'training_method': 'active',
            **results_active
        }
        final_results.append(results_active)

        # standard
        print(f"\nTraining and evaluating model using {len(labeled_indices)} randomly selected samples...")
        model_standard, _ = run_standard_finetuning(len(labeled_indices), approach, model_name, train_dataloader_batch_size, learning_rate, weight_decay, epochs, training_pool, val_dataloader, patience)
        results_standard = test_model(model_standard, test_dataloader, return_results=True)
        results_standard = {
            'changing_var': 'n_samples',
            'local_exp_idx': current_cycle-1,
            'value': len(labeled_indices),
            'training_method': 'standard',
            **results_standard
        }
        final_results.append(results_standard)
        # save to disk each time to save progress
        results_df = pd.DataFrame(final_results)
        all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)
        all_results_df.to_csv(results_path, index=False)
        print(f"Progress for experiment {current_cycle-1} appended to {results_path}")

        # if it's the last cycle, skip ensemble predictions
        if (current_cycle == total_cycles):
            print("Experiments complete.")
            break

        print("Starting ensemble training and pool evaluation...")
        ensemble_predictions = train_bootstrapped_ensemble(n_models, model_name, approach, learning_rate, weight_decay, epochs, labeled_indices, train_dataloader_batch_size, training_pool, pool_dataloader, val_dataloader, patience)
    
        current_cycle += 1
    return all_results_df