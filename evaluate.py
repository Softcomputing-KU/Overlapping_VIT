import torch
from models.lsvit import LSViT
from models.vit import ViT
from utils.data_loader import get_data_loaders
from utils.metrics import accuracy
from config import config

def evaluate(checkpoint_path):
    # Load data
    _, val_loader = get_data_loaders(config['data_dir'], config['batch_size'], config['image_size'])

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
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(config['device'])

    # Evaluation
    model.eval()
    val_acc = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(config['device']), target.to(config['device'])
            output = model(data)
            val_acc += accuracy(output, target)

    val_acc /= len(val_loader)
    print(f'Validation Accuracy: {val_acc}')

if __name__ == '__main__':
    evaluate(checkpoint_path='path_to_checkpoint.pth')