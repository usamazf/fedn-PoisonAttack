"""Training function to train the model for given number of epochs."""

import torch
import torch.nn as nn

def train_model(
        model,
        trainloader: torch.utils.data.DataLoader,
        epochs: int,
        device: str,
        learning_rate: float,
        criterion,
        optimizer,
    ) -> None:
    """Train the model."""
    # Define loss and optimizer
    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")
   
    # Train the model
    for e in range(epochs):
        running_loss = 0.0
        for b, (batch_x, batch_y) in enumerate(trainloader):
            # Prepare batch
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # Train on current batch
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            # Compute and print statistics
            running_loss += loss.item()
            if b % 100 == 0:  # print every 500 mini-batches
                print(f"Epoch {e}/{epochs-1} | Batch: {b}/{len(trainloader)-1} | Loss: {loss.item()}")
