import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.lsvit import LSViT
from models.vit import ViT
from utils.data_loader import get_data_loaders
from utils.metrics import accuracy
from config import config

def train():
    # Load data
    train_loader, val_loader = get_data_loaders(config['data_dir'], config['batch_size'], config['image_size'])

    # Initialize model
    vit_model = ViT(
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        num_classes=config['num_classes'],
        dim=config['dim'],
        depth=config['depth'],
        heads=config['heads'],
        mlp_dim=config['mlp_dim']
    )
    model = LSViT(vit_model, num_selected_tokens=config['num_selected_tokens'])
    model = model.to(config['device'])

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)

    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(config['device']), target.to(config['device'])
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')

        # Validation
        model.eval()
        val_acc = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(config['device']), target.to(config['device'])
                output = model(data)
                val_acc += accuracy(output, target)

        val_acc /= len(val_loader)
        print(f'Epoch {epoch}, Validation Accuracy: {val_acc}')

if __name__ == '__main__':
    train()