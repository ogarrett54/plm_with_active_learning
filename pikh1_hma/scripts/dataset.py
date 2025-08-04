import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class BindingDataset(Dataset):
    """
    Pytorch Dataset for protein sequences and their binding scores.
    Handles tokenization using a specified HuggingFace transformer model.
    """
    def __init__(self, sequences, scores, tok_model):
        # make sure sequence and scores have the same length
        assert len(sequences) == len(scores), f"Sequences and scores must be of the same length.\nNumber of sequences: {len(sequences)}\nNumber of scores: {len(scores)}"
        self.sequences = sequences
        self.scores = scores
        self.tokenizer = AutoTokenizer.from_pretrained(tok_model)

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self,idx):
        sequence = self.sequences[idx]
        label = torch.tensor(self.scores[idx], dtype=torch.float)

        # tokenize the sequence
        tokenized = self.tokenizer(
            sequence,
            max_length=80, # 78 residues + 2 extra tokens
            return_tensors='pt'
        )

        # return input_ids: attention masks, removing the batch dimension
        inputs = {key: val.squeeze(0) for key, val in tokenized.items()}

        return inputs, label