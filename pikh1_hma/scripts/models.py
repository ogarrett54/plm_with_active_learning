from transformers import AutoModelForSequenceClassification
import torch
from torch import nn

def get_model(approach, model_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if approach == "cls-based":
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 1).to(device)
    return model

def enable_dropout(model, prob):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = prob
            m.train()