import torch

# Tokenizer model name
TOK_MODEL = "facebook/esm2_t12_35M_UR50D"

# Data
DATA_PATH = "avrpikC_full.csv"
SEQUENCE_COL = "aa_sequence"
SCORE_COL = "enrichment_score"

# Data splitting
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# Dataloader
BATCH_SIZE = 16

# System
DEVICE = "cuda" if torch.cuda.is_available else "cpu"

# Model
APPROACH = "cls-based"
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

