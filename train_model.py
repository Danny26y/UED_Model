import argparse
import torch
from torch.utils.data import DataLoader, Subset
from dataset import UEDDataset
from model.hmlf_net import HMLFNet
from trainer import Trainer
from evaluate import evaluate_model
from export import export_model
import os

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description="Train HMLFNet for UED Detection")
        parser.add_argument('--data_dir', type=str, default='./data')
        parser.add_argument('--output_dir', type=str, default='./results')
        parser.add_argument('--epochs', type=int, default=60)
        parser.add_argument('--fast_debug', action='store_true', help="Run 2 epochs on 200 samples")
        args = parser.parse_args()
    else:
        # Default fast_debug if not present when called from main
        if not hasattr(args, 'fast_debug'):
            args.fast_debug = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    stats_file = os.path.join(args.data_dir, 'dataset_stats.json')
    train_ds = UEDDataset(os.path.join(args.data_dir, 'train.npz'), stats_file, split='train')
    val_ds = UEDDataset(os.path.join(args.data_dir, 'val.npz'), stats_file, split='val')
    test_ds = UEDDataset(os.path.join(args.data_dir, 'test.npz'), stats_file, split='test')

    if args.fast_debug:
        train_ds = Subset(train_ds, range(200))
        val_ds = Subset(val_ds, range(100))
        test_ds = Subset(test_ds, range(100))
        args.epochs = 2

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    model = HMLFNet()

    trainer = Trainer(model, train_loader, val_loader, device, epochs=args.epochs)
    best_model = trainer.fit()

    evaluate_model(best_model, test_loader, device, args.output_dir)
    export_model(best_model, './model_export')

if __name__ == '__main__':
    main()
