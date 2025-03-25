import torch
import torch.nn as nn

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super(ViT, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.dim = dim

        self.patch_embedding = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim),
            num_layers=depth
        )

        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embedding(x)  # B, dim, num_patches_h, num_patches_w
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, dim

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positional_encoding

        x = self.transformer(x)
        attention_weights = x.mean(dim=1)  # Average attention weights

        return x, attention_weights