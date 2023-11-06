"""Evaluation function to test the model performance."""

from typing import Tuple

import torch
import torch.nn as nn

def evaluate_model(
        model,
        dataloader: torch.utils.data.DataLoader,
        device: str,
    ) -> dict:
    
    """Validate the model on the entire test set."""
    # Set model to evaluation mode
    model.eval()
    
    # Evaluate model with cross entropy loss
    # TODO: upgrade to custom loss functions
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for b, (batch_x, batch_y) in enumerate(dataloader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss += criterion(outputs, batch_y).item()
            _, predicted = torch.max(outputs.data, 1)  
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    accuracy = correct / total
    
    # Return model stats on provided dataset
    return {
        "loss": loss, 
        "accuracy": accuracy,
    }