"""
head_init.py — Final layer initialization (student-implemented).

Students: Implement `init_last_layer` to control how the new classification
head is initialized before fine-tuning begins. The skeleton below uses
Kaiming uniform weights and zero bias — you are expected to experiment with
alternatives (e.g. Xavier, orthogonal, small-scale random, learned bias init).
"""

import torch
import torch.nn as nn

def init_last_layer(layer: nn.Linear) -> None:
    # Orthogonal init helps the model start with diverse class representations
    nn.init.orthogonal_(layer.weight)
    # Start with zero bias to avoid initial class bias
    nn.init.zeros_(layer.bias)
    # Scale down slightly to keep initial loss manageable
    layer.weight.data.mul_(0.1)
