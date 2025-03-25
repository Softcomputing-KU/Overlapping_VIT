config = {
    'data_dir': 'data/AID',  # Path to the dataset
    'batch_size': 8,
    'image_size': 224,
    'patch_size': 16,
    'num_classes': 30,  # Number of classes in AID dataset
    'dim': 768,  # Dimension of the Transformer
    'depth': 12,  # Number of Transformer layers
    'heads': 12,  # Number of attention heads
    'mlp_dim': 3072,  # Dimension of the MLP in Transformer
    'num_selected_tokens': 10,  # Number of tokens to select in LSViT
    'lr': 0.03,  # Learning rate
    'epochs': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}