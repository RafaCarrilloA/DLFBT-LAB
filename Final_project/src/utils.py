import random
import numpy as np
import torch

# ==========================================
# 1. REPRODUCIBILITY SETUP
# ==========================================
def set_seed(seed=42, **kwargs):
    # standard python random
    random.seed(seed)
    
    # numpy seed for splits/matrices
    np.random.seed(seed)
    
    # pytorch cpu and gpu seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # force cuDNN deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False