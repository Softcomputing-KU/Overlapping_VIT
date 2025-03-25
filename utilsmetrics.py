import torch

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = torch.eq(pred, target).float().sum().item()
        return correct / target.size(0)