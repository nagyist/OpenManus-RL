import time
import json
import openai
import random
import torch
import re
import numpy as np
from transformers import set_seed as transformers_set_seed


def refine_prompt(prompt, **kwargs):
    for key, value in kwargs.items():
        prompt = prompt.replace('$'+key, value)
    return prompt


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    transformers_set_seed(seed)


