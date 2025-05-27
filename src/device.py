"""
Utility functions for handling PyTorch devices and reproducibility
"""

import copy
import torch


def get_device(name):
    """
    Enable PyTorch with CUDA if available.
    
    :param int gpu: device number for CUDA
    :returns: device name for CUDA
    :rtype: string
    """
    if torch.cuda.is_available() is False and name[:4] == 'cuda':
        name = 'cpu'
    if torch.backends.mps.is_available() is False and name == 'mps':
        name = 'cpu'
    dev = torch.device(name)
    return dev



def set_reproducibility_PyTorch_seed(seed, deterministic=True):
    """
    Set the seed for PyTorch to control for reproducibility. 
    https://pytorch.org/docs/stable/notes/randomness.html

    Does not affect the PyTorch DataLoader multi-processing randomness
    """
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)  # includes torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = not deterministic  # causes cuDNN to deterministically select an algorithm, 
                                                        # possibly at the cost of reduced performance
