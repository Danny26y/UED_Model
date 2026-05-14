import torch
import os
from model.hmlf_net import HMLFNet
from dataset import UEDDataset
from torch.utils.data import DataLoader, Subset
from trainer import Trainer

def rebuild_model():
    # Let's train for 1 epoch to get a real state dict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stats_file = os.path.join('./data', 'dataset_stats.json')
    train_ds = UEDDataset(os.path.join('./data', 'train.npz'), stats_file, split='train')
    train_ds = Subset(train_ds, range(100))
    train_loader = DataLoader(train_ds, batch_size=32)
    val_ds = UEDDataset(os.path.join('./data', 'val.npz'), stats_file, split='val')
    val_ds = Subset(val_ds, range(32))
    val_loader = DataLoader(val_ds, batch_size=32)

    model = HMLFNet()
    trainer = Trainer(model, train_loader, val_loader, device, epochs=1)
    model = trainer.fit()

    # User requested model_export/hmlf_net.pt
    torch.save(model.state_dict(), 'model_export/hmlf_net.pt')
    print("Saved HMLFNet PyTorch checkpoint to model_export/hmlf_net.pt")

if __name__ == '__main__':
    rebuild_model()
