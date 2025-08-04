import torch
from tqdm import tqdm

def get_pool_predictions(model, pool_dataloader, device):
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

# TODO: Add acquisition functions, these will change depending on the approach