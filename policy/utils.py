import numpy as np
import random
import torch
    
def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)    # use only for multi-GPU training
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def get_rng_state(device):
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(), 
        torch.cuda.get_rng_state(device) if torch.cuda.is_available and "cuda" in str(device) else None
        )

def set_rng_state(state, device):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None: torch.cuda.set_rng_state(state[3], device)

def norm(vector):
    return np.sqrt(vector[0]**2 + vector[1]**2)