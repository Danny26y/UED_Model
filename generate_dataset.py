import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import json
import argparse
import os

def generate_mag(label):
    """Generates magnetometer data (5x5)."""
    shape = (5, 5)
    # Background Earth field baseline
    base = np.random.normal(48.0, 0.5, shape)

    # Correlated low-frequency spatial noise
    spatial_noise = scipy.ndimage.gaussian_filter(np.random.normal(0, 1, shape), sigma=1.5)

    # i.i.d. noise
    iid_noise = np.random.normal(0, 0.3, shape)

    anomaly = np.zeros(shape)
    r_c, c_c = np.nan, np.nan

    rotation_offset = 0.0

    if label == 1:
        amp = np.random.uniform(8, 20)
        sig = np.random.uniform(0.8, 1.5)
        rotation_offset = np.random.uniform(-15, 15)
        anomaly, r_c, c_c = make_dipole_anomaly(shape, amp, sig)
    elif label == 2:
        amp = np.random.uniform(1, 4)
        sig = np.random.uniform(1.0, 2.0)
        anomaly, r_c, c_c = make_dipole_anomaly(shape, amp, sig)
    elif label == 3:
        amp = np.random.uniform(0, 1.5)
        sig = np.random.uniform(1.0, 2.0) # default sig
        anomaly, r_c, c_c = make_dipole_anomaly(shape, amp, sig)

    mag = base + spatial_noise + iid_noise + anomaly + rotation_offset
    return mag, r_c, c_c

def make_dipole_anomaly(shape, amplitude, sigma):
    """Creates a 2D Gaussian dipole anomaly."""
    center_r = np.random.uniform(0, shape[0]-1)
    center_c = np.random.uniform(0, shape[1]-1)
    angle = np.random.uniform(0, 2*np.pi)

    Y, X = np.indices(shape)
    dx = X - center_c
    dy = Y - center_r

    rx = dx * np.cos(angle) - dy * np.sin(angle)
    ry = dx * np.sin(angle) + dy * np.cos(angle)

    dipole = rx * np.exp(- (rx**2 + ry**2) / (2 * sigma**2))

    max_val = np.max(np.abs(dipole))
    if max_val > 1e-6:
        dipole = dipole / max_val * amplitude

    return dipole, center_r, center_c

def generate_thermal(label):
    """Generates thermal data (8x8)."""
    shape = (8, 8)

    ambient = np.random.normal(28.0, 0.4, shape)

    # Class 0: random warm patches
    if label == 0:
        if np.random.rand() < 0.15:
            r = np.random.randint(0, shape[0])
            c = np.random.randint(0, shape[1])
            ambient[r, c] += np.random.uniform(0.5, 2.5) # resulting in ~ 28.5 to 30.5

    anomaly = np.zeros(shape)
    r_c, c_c = np.nan, np.nan

    if label == 1:
        # Metallic UED: cool elliptical anomaly surrounded by warm annulus
        r_c = np.random.uniform(1, shape[0]-2)
        c_c = np.random.uniform(1, shape[1]-2)
        Y, X = np.indices(shape)
        dist = np.sqrt((X - c_c)**2 + (Y - r_c)**2)

        delta_cool = np.random.uniform(-0.8, -0.3)
        delta_warm = np.random.uniform(0.2, 0.5)

        anomaly[dist < 1.5] = delta_cool
        annulus_mask = (dist >= 1.5) & (dist < 2.5)
        anomaly[annulus_mask] = delta_warm
        anomaly = scipy.ndimage.gaussian_filter(anomaly, sigma=0.5)

    elif label == 2:
        # Plastic UED: slightly warm anomaly, soft-edged
        r_c = np.random.uniform(1, shape[0]-2)
        c_c = np.random.uniform(1, shape[1]-2)
        Y, X = np.indices(shape)
        dist = np.sqrt((X - c_c)**2 + (Y - r_c)**2)

        delta = np.random.uniform(0.5, 1.5)
        anomaly[dist < 2.0] = delta
        anomaly = scipy.ndimage.gaussian_filter(anomaly, sigma=1.0)

    elif label == 3:
        # IED/AN: moderate warm anomaly, irregular shape
        r_c = np.random.uniform(1, shape[0]-2)
        c_c = np.random.uniform(1, shape[1]-2)
        Y, X = np.indices(shape)
        dist = np.sqrt((X - c_c)**2 + (Y - r_c)**2)

        delta = np.random.uniform(0.8, 2.0)
        blob = np.exp(-dist**2 / (2 * 1.5**2))
        perturbation = np.random.uniform(0.5, 1.5, shape)
        anomaly = blob * perturbation * delta

    iid_noise = np.random.normal(0, 0.2, shape)

    crossover = np.zeros(shape)
    if np.random.rand() < 0.08:
        # random 2-pixel cold stripe
        row = np.random.randint(0, shape[0])
        col = np.random.randint(0, shape[1]-1)
        crossover[row, col:col+2] = -1.0 # adjust arbitrary cold delta

    thermal = ambient + anomaly + iid_noise + crossover
    return thermal, r_c, c_c

def generate_ar1(length, phi, std):
    """Generates AR(1) temporal noise."""
    noise = np.zeros(length)
    noise[0] = np.random.normal(0, std)
    for i in range(1, length):
        noise[i] = phi * noise[i-1] + np.random.normal(0, std)
    return noise

def generate_gas(label):
    """Generates gas data (20, 2)."""
    length = 20

    mq2_base = np.random.normal(4.5, 0.3)
    mq135_base = np.random.normal(3.8, 0.25)

    mq2 = np.full(length, mq2_base)
    mq135 = np.full(length, mq135_base)

    mq2_ar1 = generate_ar1(length, 0.7, 0.1) # generic std for wind drift, maybe scale?
    mq135_ar1 = generate_ar1(length, 0.7, 0.1)

    if label == 1:
        pass # Same as Class 0
    elif label == 2:
        # MQ-135 gradual descent
        descent = np.random.uniform(0.6, 1.4)
        mq135 -= np.linspace(0, descent, length)
    elif label == 3:
        # Both descend, randomized start frame
        onset = int(np.random.uniform(0, 8))
        descent_mq2 = np.random.uniform(0.4, 1.0)
        descent_mq135 = np.random.uniform(0.8, 1.8)

        rem_len = length - onset
        mq2[onset:] -= np.linspace(0, descent_mq2, rem_len)
        mq135[onset:] -= np.linspace(0, descent_mq135, rem_len)

    mq2 += mq2_ar1 + np.random.normal(0, 0.05, length)
    mq135 += mq135_ar1 + np.random.normal(0, 0.05, length)

    # Simulate saturation
    mq2 = np.clip(mq2, 0.05, 10.0)
    mq135 = np.clip(mq135, 0.05, 10.0)

    gas = np.stack([mq2, mq135], axis=-1)
    return gas


def main():
    """Main generation loop for the UED dataset."""
    parser = argparse.ArgumentParser(description="Generate UED dataset")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    samples_per_class = 5000
    classes = 4
    total_samples = samples_per_class * classes

    # Pre-allocate arrays
    mag_data = np.zeros((total_samples, 5, 5), dtype=np.float32)
    thermal_data = np.zeros((total_samples, 8, 8), dtype=np.float32)
    gas_data = np.zeros((total_samples, 20, 2), dtype=np.float32)
    labels = np.zeros(total_samples, dtype=np.int64)
    metadata = np.zeros((total_samples, 4), dtype=np.float32) # r_mag, c_mag, r_therm, c_therm

    idx = 0
    for c in range(classes):
        for _ in range(samples_per_class):
            m, rm, cm = generate_mag(c)
            t, rt, ct = generate_thermal(c)
            g = generate_gas(c)

            mag_data[idx] = m
            thermal_data[idx] = t
            gas_data[idx] = g
            labels[idx] = c

            # Nan for safe class handled in generation logic partially, let's explicitly set
            if c == 0:
                metadata[idx] = [np.nan, np.nan, np.nan, np.nan]
            else:
                metadata[idx] = [rm, cm, rt, ct]

            idx += 1

    # Stratified split 70 / 15 / 15
    train_indices = []
    val_indices = []
    test_indices = []

    for c in range(classes):
        start = c * samples_per_class
        end = start + samples_per_class
        indices = np.arange(start, end)
        np.random.shuffle(indices)

        n_train = int(0.7 * samples_per_class)
        n_val = int(0.15 * samples_per_class)

        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train+n_val])
        test_indices.extend(indices[n_train+n_val:])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples: {len(val_indices)}")
    print(f"Test samples: {len(test_indices)}")

    calculate_stats(mag_data, thermal_data, gas_data, labels, args)

    save_npz_and_plot(mag_data, thermal_data, gas_data, labels, metadata, train_indices, val_indices, test_indices, args)

    return mag_data, thermal_data, gas_data, labels, metadata, train_indices, val_indices, test_indices, args

def save_npz_and_plot(mag_data, thermal_data, gas_data, labels, metadata, train_indices, val_indices, test_indices, args):
    """Saves data to NPZ files and generates a preview plot."""
    print("\nSaving NPZ files...")
    np.savez_compressed(
        os.path.join(args.output_dir, 'train.npz'),
        mag=mag_data[train_indices],
        thermal=thermal_data[train_indices],
        gas=gas_data[train_indices],
        labels=labels[train_indices],
        metadata=metadata[train_indices]
    )

    np.savez_compressed(
        os.path.join(args.output_dir, 'val.npz'),
        mag=mag_data[val_indices],
        thermal=thermal_data[val_indices],
        gas=gas_data[val_indices],
        labels=labels[val_indices],
        metadata=metadata[val_indices]
    )

    np.savez_compressed(
        os.path.join(args.output_dir, 'test.npz'),
        mag=mag_data[test_indices],
        thermal=thermal_data[test_indices],
        gas=gas_data[test_indices],
        labels=labels[test_indices],
        metadata=metadata[test_indices]
    )

    print("Generating data preview plot...")
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    fig.suptitle('Underground Explosive Device (UED) Dataset - Sample Preview', fontsize=16)

    class_names = ['0: Safe', '1: Metallic UED', '2: Plastic UED', '3: IED/AN']

    for c in range(4):
        mask = np.where(labels == c)[0]
        idx = np.random.choice(mask)

        # Mag
        im0 = axes[c, 0].imshow(mag_data[idx], cmap='viridis')
        axes[c, 0].set_title(f'Mag: Class {class_names[c]}')
        fig.colorbar(im0, ax=axes[c, 0])

        # Thermal
        im1 = axes[c, 1].imshow(thermal_data[idx], cmap='inferno')
        axes[c, 1].set_title(f'Thermal: Class {class_names[c]}')
        fig.colorbar(im1, ax=axes[c, 1])

        # Gas
        axes[c, 2].plot(gas_data[idx, :, 0], label='MQ-2', color='blue')
        axes[c, 2].plot(gas_data[idx, :, 1], label='MQ-135', color='red')
        axes[c, 2].set_title(f'Gas: Class {class_names[c]}')
        axes[c, 2].set_ylim(0, 10)
        axes[c, 2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'data_preview.png'))
    plt.close()


def calculate_stats(mag_data, thermal_data, gas_data, labels, args):
    """Calculates summary statistics and saves to json."""
    classes = 4
    stats = {}

    for c in range(classes):
        mask = (labels == c)
        c_mag = mag_data[mask]
        c_therm = thermal_data[mask]
        c_gas = gas_data[mask]

        stats[str(c)] = {
            "mag": {
                "mean": float(np.mean(c_mag)),
                "std": float(np.std(c_mag)),
                "min": float(np.min(c_mag)),
                "max": float(np.max(c_mag))
            },
            "thermal": {
                "mean": float(np.mean(c_therm)),
                "std": float(np.std(c_therm)),
                "min": float(np.min(c_therm)),
                "max": float(np.max(c_therm))
            },
            "gas_mq2": {
                "mean": float(np.mean(c_gas[:, :, 0])),
                "std": float(np.std(c_gas[:, :, 0])),
                "min": float(np.min(c_gas[:, :, 0])),
                "max": float(np.max(c_gas[:, :, 0]))
            },
            "gas_mq135": {
                "mean": float(np.mean(c_gas[:, :, 1])),
                "std": float(np.std(c_gas[:, :, 1])),
                "min": float(np.min(c_gas[:, :, 1])),
                "max": float(np.max(c_gas[:, :, 1]))
            }
        }

    print("\nGeneration Summary:")
    print("="*80)
    print(f"{'Class':<6} | {'Mag Range':<20} | {'Thermal Range':<20} | {'Gas MQ-2 Range':<15}")
    print("-" * 80)
    for c in range(classes):
        mag_min, mag_max = stats[str(c)]['mag']['min'], stats[str(c)]['mag']['max']
        thm_min, thm_max = stats[str(c)]['thermal']['min'], stats[str(c)]['thermal']['max']
        g2_min, g2_max = stats[str(c)]['gas_mq2']['min'], stats[str(c)]['gas_mq2']['max']
        print(f"{c:<6} | {mag_min:6.2f} to {mag_max:6.2f} | {thm_min:6.2f} to {thm_max:6.2f} | {g2_min:6.2f} to {g2_max:6.2f}")
    print("="*80)

    with open(os.path.join(args.output_dir, 'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)

if __name__ == "__main__":
    main()
