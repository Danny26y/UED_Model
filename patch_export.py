import torch
import os
import json
from model.hmlf_net import HMLFNet

def patch_save():
    model = HMLFNet()
    # Assume we just need to save the model from the unquantised state if it exists,
    # but we can also just run the fast_debug training again.
