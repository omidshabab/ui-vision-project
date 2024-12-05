import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from models.uinet import UIDetectionNet
from data.dataset import UIDataset
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(epochs=100):
    logger.info("Starting training...")
    
    # Check for data directories
    if not os.path.exists('data/images') or not os.path.exists('data/annotations'):
        raise RuntimeError("Data directories not found. Create data/images and data/annotations first.")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Load dataset
    try:
        dataset = UIDataset(
            'data/images',
            'data/annotations',
            transform=transform
        )
        logger.info(f"Dataset loaded with {len(dataset)} samples")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {str(e)}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )
    
    # Initialize model and optimizer
    model = UIDetectionNet().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, annotations) in enumerate(dataloader):
            images = images.to(device)
            annotations = annotations.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, annotations)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f'Epoch {epoch}: Average Loss = {avg_loss:.4f}')
        
        # Save model
        torch.save(model.state_dict(), 'models/trained_model.pth')
        logger.info(f"Model saved to models/trained_model.pth")

if __name__ == "__main__":
    train()