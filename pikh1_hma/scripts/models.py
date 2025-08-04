from transformers import AutoModelForSequenceClassification

def get_model(approach, model_name):
    if approach == "cls-based":
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 1)
    return model