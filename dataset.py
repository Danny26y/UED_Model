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

        # Let's read stats from json. We will compute the global pooled variance
        # from the class means and variances, assuming equal sized classes.

        def compute_pooled_stats(means, stds):
            means = np.array(means)
            stds = np.array(stds)
            variances = stds ** 2
            global_mean = np.mean(means)
            # Global variance = mean of variances + variance of means
            global_var = np.mean(variances) + np.var(means)
            return global_mean, np.sqrt(global_var)

        active_classes = [0, 1, 3]

        mag_means = [stats[str(c)]['mag']['mean'] for c in active_classes]
        mag_stds = [stats[str(c)]['mag']['std'] for c in active_classes]
        self.mag_mean, self.mag_std = compute_pooled_stats(mag_means, mag_stds)

        thm_means = [stats[str(c)]['thermal']['mean'] for c in active_classes]
        thm_stds = [stats[str(c)]['thermal']['std'] for c in active_classes]
        self.thm_mean, self.thm_std = compute_pooled_stats(thm_means, thm_stds)

        g2_means = [stats[str(c)]['gas_mq2']['mean'] for c in active_classes]
        g2_stds = [stats[str(c)]['gas_mq2']['std'] for c in active_classes]
        self.g2_mean, self.g2_std = compute_pooled_stats(g2_means, g2_stds)

        g135_means = [stats[str(c)]['gas_mq135']['mean'] for c in active_classes]
        g135_stds = [stats[str(c)]['gas_mq135']['std'] for c in active_classes]
        self.g135_mean, self.g135_std = compute_pooled_stats(g135_means, g135_stds)

        # Filter out Class 2
        valid_indices = [i for i, lbl in enumerate(self.labels) if lbl != 2]
        self.mag = self.mag[valid_indices]
        self.thermal = self.thermal[valid_indices]
        self.gas = self.gas[valid_indices]
        self.labels = self.labels[valid_indices]
        self.metadata = self.metadata[valid_indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mag = self.mag[idx].copy()
        thermal = self.thermal[idx].copy()
        gas = self.gas[idx].copy()
        label = self.labels[idx]

        # Remap Class 3 to Class 2 to maintain contiguous 0,1,2 range
        if label == 3:
            label = 2

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
                        # np.rot90 rotates counter-clockwise.
                        # Original points: (0,0)->(4,0), (0,4)->(0,0), (4,0)->(4,4), (4,4)->(0,4)
                        # The CCW rotation formula for an NxN grid (here N=5 so max index is 4):
                        # new_r = 4 - c
                        # new_c = r
                        new_r = 4 - c
                        new_c = r
                        r, c = new_r, new_c
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
