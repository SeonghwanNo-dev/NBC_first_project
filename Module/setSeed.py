import torch
import numpy as np
import random
import os

def set_seed(seed_value):
    """모든 난수 발생원에 시드를 설정하여 재현성을 확보합니다."""
    
    # 1. Python, NumPy 시드 설정
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    # 2. PyTorch 시드 설정 (CPU 및 CUDA)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    
    # 3. 추가 설정 (CUDA 결정론적 동작 보장)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    
    
    print(f"Random seed set to {seed_value}")