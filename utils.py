import torch

def cycle_consistency_loss(reconstructed, original, criterion, lambda_weight):
    """
    Computes the cycle-consistency loss between the original and reconstructed images.

    Parameters:
        - reconstructed: the reconstructed image (output of generator network)
        - original: the original image (input to generator network)
        - criterion: loss function to compute the cycle-consistency loss
        - lambda_weight: weight to scale the cycle-consistency loss (hyperparameter)

    Returns:
        - cycle-consistency loss value
    """
    return lambda_weight * criterion(reconstructed, original)

def identity_loss(real_images, generated_images, lambda_weight):
    """Calculate identity loss for the generator"""
    identity_loss = lambda_weight * torch.mean(torch.abs(real_images - generated_images))
    return identity_loss

def adversarial_loss(y_pred, y_true, criterion):
    """
    Computes the adversarial loss for a binary classification task.
    
    Args:
        y_pred (torch.Tensor): The predicted output of the discriminator.
        y_true (bool): The true label for the input data.
        criterion: The loss function used to compute the adversarial loss.
        
    Returns:
        The adversarial loss.
    """
    target = torch.full_like(y_pred, y_true, device=y_pred.device)
    return criterion(y_pred, target)