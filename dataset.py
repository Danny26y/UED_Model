import torch
from torch.utils.data import Dataset
import numpy as np
import json

class UEDDataset(Dataset):
    def __init__(self, npz_file, stats_file, split='train'):
        super().__init__()
        data = np.load(npz_file)
        self.mag = data['mag']
        self.thermal = data['thermal']
        self.gas = data['gas']
        self.labels = data['labels']
        self.metadata = data['metadata']

        self.split = split

        with open(stats_file, 'r') as f:
            stats = json.load(f)

        # Z-score normalization requires global mean/std per modality, but we were only given per-class stats.
        # We need to compute global stats from the per-class stats, or recompute them from the loaded data.
        # The prompt says: "Normalise all inputs using stats from `dataset_stats.json` (Z-score normalisation per modality)"
        # Since we just have the data, we can recompute global stats directly from the training split, or reconstruct.
        # Actually, let's just compute from the loaded data for simplicity and correctness.
        # Wait, the prompt specifically says "using stats from dataset_stats.json".

        # Let's read stats from json. We will average the means and stds across the 4 classes.
        # This is an approximation of the global mean/std, assuming balanced classes.

        mag_means = [stats[str(c)]['mag']['mean'] for c in range(4)]
        mag_stds = [stats[str(c)]['mag']['std'] for c in range(4)]
        self.mag_mean = np.mean(mag_means)
        self.mag_std = np.mean(mag_stds)

        thm_means = [stats[str(c)]['thermal']['mean'] for c in range(4)]
        thm_stds = [stats[str(c)]['thermal']['std'] for c in range(4)]
        self.thm_mean = np.mean(thm_means)
        self.thm_std = np.mean(thm_stds)

        g2_means = [stats[str(c)]['gas_mq2']['mean'] for c in range(4)]
        g2_stds = [stats[str(c)]['gas_mq2']['std'] for c in range(4)]
        self.g2_mean = np.mean(g2_means)
        self.g2_std = np.mean(g2_stds)

        g135_means = [stats[str(c)]['gas_mq135']['mean'] for c in range(4)]
        g135_stds = [stats[str(c)]['gas_mq135']['std'] for c in range(4)]
        self.g135_mean = np.mean(g135_means)
        self.g135_std = np.mean(g135_stds)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mag = self.mag[idx].copy()
        thermal = self.thermal[idx].copy()
        gas = self.gas[idx].copy()
        label = self.labels[idx]
        meta = self.metadata[idx].copy()

        if self.split == 'train':
            # Data augmentation
            # Mag: Random Gaussian noise (std=0.1 µT), random 90° rotation
            mag += np.random.normal(0, 0.1, mag.shape).astype(np.float32)
            rotations = np.random.randint(0, 4)
            if rotations > 0:
                mag = np.rot90(mag, k=rotations)
                # Update metadata targets for Mag (row, col are index 0, 1)
                if not np.isnan(meta[0]):
                    r, c = meta[0], meta[1]
                    for _ in range(rotations):
                        r, c = c, 4 - r # 90 degree counter-clockwise rotation matching np.rot90
                    meta[0], meta[1] = r, c

            # Thermal: Random Gaussian noise (std=0.15°C), random horizontal flip
            thermal += np.random.normal(0, 0.15, thermal.shape).astype(np.float32)
            if np.random.rand() > 0.5:
                thermal = np.fliplr(thermal)
                # Update metadata targets for Thermal (row, col are index 2, 3)
                if not np.isnan(meta[2]):
                    meta[3] = 7 - meta[3] # fliplr flips columns

            # Gas: Random time-shift (roll by ±3 frames), random amplitude scale U(0.9, 1.1)
            shift = np.random.randint(-3, 4)
            gas = np.roll(gas, shift=shift, axis=0)
            scale = np.random.uniform(0.9, 1.1)
            gas *= scale

        # Normalization
        mag = (mag - self.mag_mean) / (self.mag_std + 1e-8)
        thermal = (thermal - self.thm_mean) / (self.thm_std + 1e-8)
        gas[:, 0] = (gas[:, 0] - self.g2_mean) / (self.g2_std + 1e-8)
        gas[:, 1] = (gas[:, 1] - self.g135_mean) / (self.g135_std + 1e-8)

        # Expand dims for image models
        mag = np.expand_dims(mag, axis=0) # (1, 5, 5)
        thermal = np.expand_dims(thermal, axis=0) # (1, 8, 8)

        return {
            'mag': torch.tensor(mag, dtype=torch.float32),
            'thermal': torch.tensor(thermal, dtype=torch.float32),
            'gas': torch.tensor(gas, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'metadata': torch.tensor(meta, dtype=torch.float32)
        }
