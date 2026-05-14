import argparse
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.animation as animation
from dataset import UEDDataset
from model.hmlf_net import HMLFNet
import time

plt.style.use('dark_background')
os.makedirs('visualisations', exist_ok=True)

def load_data_and_model(sample_idx=-1, random=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stats_file = './data/dataset_stats.json'
    test_ds = UEDDataset('./data/test.npz', stats_file, split='test')

    if random:
        sample_idx = np.random.randint(0, len(test_ds))
    elif sample_idx < 0 or sample_idx >= len(test_ds):
        print(f"Invalid sample idx. Total samples: {len(test_ds)}")
        sample_idx = 0

    # Get raw and normalized
    # The UEDDataset currently returns normalized data.
    # To get raw, we can manually apply inverse scaling, or load it raw.
    raw_mag = test_ds.mag[sample_idx]
    raw_thm = test_ds.thermal[sample_idx]
    raw_gas = test_ds.gas[sample_idx]

    sample = test_ds[sample_idx]
    norm_mag = sample['mag'].unsqueeze(0).to(device)
    norm_thm = sample['thermal'].unsqueeze(0).to(device)
    norm_gas = sample['gas'].unsqueeze(0).to(device)
    label = sample['label'].item()
    meta = sample['metadata'].numpy()

    model = HMLFNet()
    model.load_state_dict(torch.load('model_export/hmlf_net.pt', map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits, loc, conf, gate_weights = model(norm_mag, norm_thm, norm_gas)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        preds = torch.argmax(logits, dim=1).item()
        loc = loc.cpu().numpy()[0]
        conf = conf.cpu().item()
        gate_weights = gate_weights.cpu().numpy()[0]

    return {
        'idx': sample_idx,
        'raw_mag': raw_mag,
        'raw_thm': raw_thm,
        'raw_gas': raw_gas,
        'norm_mag': norm_mag.cpu().numpy()[0, 0],
        'norm_thm': norm_thm.cpu().numpy()[0, 0],
        'label': label,
        'meta': meta,
        'probs': probs,
        'pred_class': preds,
        'loc': loc,
        'conf': conf,
        'gate_weights': gate_weights,
        'test_ds': test_ds,
        'model': model,
        'device': device
    }

import json

def panel_1_2d_heatmaps(data):
    idx = data['idx']
    raw_mag = data['raw_mag']
    raw_thm = data['raw_thm']
    raw_gas = data['raw_gas']
    meta = data['meta']
    loc = data['loc']
    gate_weights = data['gate_weights']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1a. Mag
    im0 = axes[0].imshow(raw_mag, cmap='RdBu_r')
    fig.colorbar(im0, ax=axes[0])
    axes[0].set_title(f"Magnetometer Array (µT)\nGate confidence: {gate_weights[0]:.2f}")
    if not np.isnan(meta[0]):
        axes[0].plot(meta[1], meta[0], 'go', markersize=10, markerfacecolor='none', label='True')
    if loc[0] > 0.01 or loc[1] > 0.01: # proxy for predicting non-zero location
        axes[0].plot(loc[1]*4.0, loc[0]*4.0, 'wx', markersize=10, label='Pred')
    axes[0].legend(loc='upper right')

    # 1b. Thermal
    im1 = axes[1].imshow(raw_thm, cmap='hot')
    fig.colorbar(im1, ax=axes[1])
    axes[1].set_title(f"Thermal Array (°C)\nGate confidence: {gate_weights[1]:.2f}")
    if not np.isnan(meta[2]):
        axes[1].plot(meta[3], meta[2], 'go', markersize=10, markerfacecolor='none', label='True')
    if loc[2] > 0.01 or loc[3] > 0.01:
        axes[1].plot(loc[3]*7.0, loc[2]*7.0, 'wx', markersize=10, label='Pred')
    axes[1].legend(loc='upper right')

    # 1c. Gas
    with open('./data/dataset_stats.json', 'r') as f:
        stats = json.load(f)
    baseline_mq2 = stats['0']['gas_mq2']['mean']
    std_mq2 = stats['0']['gas_mq2']['std']
    baseline_mq135 = stats['0']['gas_mq135']['mean']
    std_mq135 = stats['0']['gas_mq135']['std']

    x = np.arange(20)
    axes[2].plot(x, raw_gas[:, 0], 'b-', label='MQ-2')
    axes[2].plot(x, raw_gas[:, 1], 'r--', label='MQ-135')
    axes[2].fill_between(x, baseline_mq2 - std_mq2, baseline_mq2 + std_mq2, color='blue', alpha=0.2)
    axes[2].fill_between(x, baseline_mq135 - std_mq135, baseline_mq135 + std_mq135, color='red', alpha=0.2)
    axes[2].set_title(f"Gas Sensors (Rs/Ro)\nGate confidence: {gate_weights[2]:.2f}")
    axes[2].set_xlabel("Frame index")
    axes[2].set_ylabel("Rs/Ro")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f'visualisations/2d_inputs_sample_{idx}.png')
    plt.show()

def panel_4_decision_dashboard(data, fuzzy_status):
    idx = data['idx']
    probs = data['probs']
    gate_weights = data['gate_weights']
    norm_mag = data['norm_mag']
    norm_thm = data['norm_thm']
    conf = data['conf']
    pred_class = data['pred_class']
    label = data['label']

    class_names = ['Safe (0)', 'Metallic UED (1)', 'Plastic UED (2)', 'IED/AN (3)']

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # [0,0] Class Probabilities
    bars = axes[0, 0].barh(class_names, probs, color='grey')
    bars[pred_class].set_color('red')
    axes[0, 0].axvline(probs[label] if label == pred_class else 0, color='green', linestyle='--', label=f'True: {class_names[label]}')
    axes[0, 0].set_title("Classification Probabilities")
    axes[0, 0].legend()

    # [0,1] Modality Gate Weights
    axes[0, 1].pie(gate_weights, labels=['Mag', 'Thermal', 'Gas'], colors=['blue', 'orange', 'green'], autopct='%1.1f%%')
    axes[0, 1].set_title("Modality Trust Weights")

    # [1,0] Normalised Mag
    im2 = axes[1, 0].imshow(norm_mag, cmap='coolwarm')
    fig.colorbar(im2, ax=axes[1, 0])
    axes[1, 0].set_title("Normalised Mag Input (model view)")

    # [1,1] Normalised Thermal
    im3 = axes[1, 1].imshow(norm_thm, cmap='coolwarm')
    fig.colorbar(im3, ax=axes[1, 1])
    axes[1, 1].set_title("Normalised Thermal Input (model view)")

    # [2,0] Confidence Gauge
    axes[2, 0].barh(['Confidence'], [1.0], color='green', alpha=0.3)
    axes[2, 0].barh(['Confidence'], [conf], color='red')
    axes[2, 0].set_xlim(0, 1)

    match_str = "✓" if pred_class == label else "✗"
    axes[2, 0].text(0.5, 0.5, f"Model Confidence: {conf:.2f}\nPred: {class_names[pred_class]}\nTrue: {class_names[label]} {match_str}",
                    ha='center', va='center', fontsize=12, transform=axes[2, 0].transAxes, color='white',
                    bbox=dict(facecolor='black', alpha=0.5))
    axes[2, 0].set_title("Decision Confidence")

    # [2,1] Fuzzy Logic Hazard Status
    ev_mag = gate_weights[0] * np.max(probs)
    ev_thm = gate_weights[1] * np.max(probs)
    ev_gas = gate_weights[2] * np.max(probs)

    axes[2, 1].barh(['Chemical', 'Thermal', 'Magnetic'], [ev_gas, ev_thm, ev_mag], color=['green', 'orange', 'blue'])
    axes[2, 1].set_xlim(0, 1)

    color = 'yellow' if 'ALERT' in fuzzy_status else 'red' if 'DANGER' in fuzzy_status else 'green'
    axes[2, 1].text(0.5, -0.5, fuzzy_status, ha='center', va='center', fontsize=24, fontweight='bold', color=color, transform=axes[2, 1].transAxes)
    axes[2, 1].set_title("Fuzzy Hazard Assessment")

    plt.tight_layout()
    plt.savefig(f'visualisations/decision_dashboard_sample_{idx}.png')
    plt.show()

def panel_2_3_plotly(data):
    idx = data['idx']
    raw_mag = data['raw_mag']
    raw_thm = data['raw_thm']
    loc = data['loc']

    # Panel 2: Mag 3D
    fig_mag = go.Figure(data=[go.Surface(z=raw_mag, colorscale='RdBu')])

    # Add predicted anomaly scatter point
    pred_r_mag = loc[0] * 4.0
    pred_c_mag = loc[1] * 4.0

    if pred_r_mag > 0.1 or pred_c_mag > 0.1: # non-zero proxy
        z_val = raw_mag[int(round(min(4, max(0, pred_r_mag)))), int(round(min(4, max(0, pred_c_mag))))]
        fig_mag.add_trace(go.Scatter3d(x=[pred_c_mag], y=[pred_r_mag], z=[z_val],
                                       mode='markers', marker=dict(size=10, color='gold', symbol='diamond')))

    fig_mag.update_layout(title="3D Magnetic Field Distribution", scene=dict(
        xaxis_title='Column', yaxis_title='Row', zaxis_title='µT'
    ), template='plotly_dark')

    fig_mag.write_html(f"visualisations/3d_magnetic_sample_{idx}.html")
    fig_mag.write_image(f"visualisations/3d_magnetic_sample_{idx}.png")
    fig_mag.show()

    # Panel 3: Thermal 3D
    fig_thm = go.Figure(data=[go.Surface(z=raw_thm, colorscale='thermal')])

    pred_r_thm = loc[2] * 7.0
    pred_c_thm = loc[3] * 7.0

    if pred_r_thm > 0.1 or pred_c_thm > 0.1:
        z_val_thm = raw_thm[int(round(min(7, max(0, pred_r_thm)))), int(round(min(7, max(0, pred_c_thm))))]
        fig_thm.add_trace(go.Scatter3d(x=[pred_c_thm], y=[pred_r_thm], z=[z_val_thm],
                                       mode='markers', marker=dict(size=10, color='gold', symbol='diamond')))

    fig_thm.update_layout(title="3D Thermal Surface Distribution", scene=dict(
        xaxis_title='Column', yaxis_title='Row', zaxis_title='°C'
    ), template='plotly_dark')

    fig_thm.write_html(f"visualisations/3d_thermal_sample_{idx}.html")
    fig_thm.write_image(f"visualisations/3d_thermal_sample_{idx}.png")
    fig_thm.show()

def animate_batch(data):
    test_ds = data['test_ds']
    model = data['model']
    device = data['device']

    # Randomly pick 30 samples (10 safe, 20 non-safe)
    safe_indices = np.where(test_ds.labels == 0)[0]
    nonsafe_indices = np.where(test_ds.labels != 0)[0]

    selected_safe = np.random.choice(safe_indices, 10, replace=False)
    selected_nonsafe = np.random.choice(nonsafe_indices, 20, replace=False)

    batch_indices = np.concatenate([selected_safe, selected_nonsafe])
    np.random.shuffle(batch_indices)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    class_names = ['Safe', 'Metallic', 'Plastic', 'IED/AN']

    def update(frame):
        idx = batch_indices[frame]
        sample = test_ds[idx]

        raw_mag = test_ds.mag[idx]
        raw_thm = test_ds.thermal[idx]

        norm_mag = sample['mag'].unsqueeze(0).to(device)
        norm_thm = sample['thermal'].unsqueeze(0).to(device)
        norm_gas = sample['gas'].unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _, _, _ = model(norm_mag, norm_thm, norm_gas)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = np.argmax(probs)

        axes[0].clear()
        axes[1].clear()
        axes[2].clear()

        axes[0].imshow(raw_mag, cmap='RdBu_r')
        axes[0].set_title("Mag Scan")

        axes[1].imshow(raw_thm, cmap='hot')
        axes[1].set_title("Thermal Scan")

        bars = axes[2].barh(class_names, probs, color='grey')
        bars[pred_class].set_color('red')
        axes[2].set_xlim(0, 1)
        axes[2].set_title("Live Prediction")

        status_text = f"PROCESSING... -> CLASSIFIED: {class_names[pred_class]}"
        fig.suptitle(status_text, fontsize=16, color='red' if pred_class != 0 else 'green')

    ani = animation.FuncAnimation(fig, update, frames=30, interval=500)
    ani.save('visualisations/batch_scan_animation.gif', writer='pillow')
    plt.show()

def print_summary(data):
    class_names = ['Safe', 'Metallic UED', 'Plastic UED', 'IED/AN']

    idx = data['idx']
    true_class = class_names[data['label']]
    pred_class = class_names[data['pred_class']]
    match_str = "✓" if data['label'] == data['pred_class'] else "✗"
    conf = data['conf']

    gw_mag, gw_thm, gw_gas = data['gate_weights']

    loc = data['loc']
    meta = data['meta']

    if data['label'] != 0:
        true_r_mag, true_c_mag = meta[0], meta[1]
        pred_r_mag, pred_c_mag = loc[0] * 4.0, loc[1] * 4.0
        dist_mag = np.sqrt((pred_r_mag - true_r_mag)**2 + (pred_c_mag - true_c_mag)**2)

        true_r_thm, true_c_thm = meta[2], meta[3]
        pred_r_thm, pred_c_thm = loc[2] * 7.0, loc[3] * 7.0
        dist_thm = np.sqrt((pred_r_thm - true_r_thm)**2 + (pred_c_thm - true_c_thm)**2)

        loc_str = f"Localisation Error -> Mag: {dist_mag:.1f} grid units | Thermal: {dist_thm:.1f} grid units"
    else:
        loc_str = "Localisation Error -> N/A (Safe class)"

    if conf > 0.7:
        fuzzy_status = "🔴 DANGER"
    elif conf >= 0.4:
        fuzzy_status = "⚠ ALERT"
    else:
        if data['pred_class'] == 0:
            fuzzy_status = "✅ SAFE"
        else:
            fuzzy_status = "⚠ ALERT" # low conf but not safe prediction

    print(f"Sample #{idx} | True: {true_class} | Predicted: {pred_class} {match_str} | Confidence: {conf:.2f}")
    print(f"Gate Weights -> Mag: {gw_mag:.2f} | Thermal: {gw_thm:.2f} | Gas: {gw_gas:.2f}")
    print(loc_str)
    print(f"Hazard Status: {fuzzy_status}")

    return fuzzy_status

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description="Interactive Visualisation Dashboard")
        parser.add_argument('--sample_idx', type=int, default=42)
        parser.add_argument('--random', action='store_true')
        # adding a switch for animation only logic
        parser.add_argument('--animate_only', action='store_true', help="Only run batch animation")
        args = parser.parse_args()
    else:
        if not hasattr(args, 'animate_only'):
            args.animate_only = False

    data = load_data_and_model(getattr(args, 'sample_idx', 42), getattr(args, 'random', False))

    if args.animate_only:
        print("Generating Panel 5 (Batch Animation)...")
        animate_batch(data)
        print("Done. Animation saved in visualisations/batch_scan_animation.gif")
        return

    fuzzy_status = print_summary(data)

    print("Generating Panel 1 (2D Heatmaps)...")
    panel_1_2d_heatmaps(data)

    print("Generating Panel 2 and 3 (3D Plotly Surfaces)...")
    panel_2_3_plotly(data)

    print("Generating Panel 4 (Decision Dashboard)...")
    panel_4_decision_dashboard(data, fuzzy_status)

    # We optionally can run animate_batch here, but prompt separates it as a subcommand. Let's include it if not animate_only but standard run.
    # Actually if standard run, we will generate everything.
    print("Generating Panel 5 (Batch Animation)...")
    animate_batch(data)

    print("Done. Visualisations saved in visualisations/")

if __name__ == '__main__':
    main()
