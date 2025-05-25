# Ghiotto Andrea   2118418

import torch
import os

def save_state(epoch, model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            }, path)
    
def load_state(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    return model