import torch
def mit_loss(imputed, original, mask):
    """Calculate the loss for the masked imputation task."""
    return torch.sum(torch.abs(imputed - original) * mask) / (torch.sum(mask)+1e-9)

import torch
def ort_loss(reconstructed, original, mask):
    """Calculate the loss for the observed reconstruction task."""
    return torch.sum(torch.abs(reconstructed - original) * mask) / (torch.sum(mask)+1e-9)