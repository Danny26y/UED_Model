import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import time

def evaluate_model(model, test_loader, device, output_dir):
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    all_gate_weights = []

    loc_errors = []

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch in test_loader:
            mag = batch['mag'].to(device)
            thermal = batch['thermal'].to(device)
            gas = batch['gas'].to(device)
            labels = batch['label'].to(device)
            meta = batch['metadata'].to(device)

            logits, loc, conf, gate_weights = model(mag, thermal, gas)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
            all_gate_weights.extend(gate_weights.cpu().numpy())

            # Localisation error calculation
            for i in range(len(labels)):
                if labels[i] != 0:
                    pred_r_mag = loc[i, 0].item() * 4.0
                    pred_c_mag = loc[i, 1].item() * 4.0
                    pred_r_thm = loc[i, 2].item() * 7.0
                    pred_c_thm = loc[i, 3].item() * 7.0

                    true_r_mag = meta[i, 0].item()
                    true_c_mag = meta[i, 1].item()
                    true_r_thm = meta[i, 2].item()
                    true_c_thm = meta[i, 3].item()

                    dist_mag = np.sqrt((pred_r_mag - true_r_mag)**2 + (pred_c_mag - true_c_mag)**2)
                    dist_thm = np.sqrt((pred_r_thm - true_r_thm)**2 + (pred_c_thm - true_c_thm)**2)

                    loc_errors.append((dist_mag + dist_thm) / 2.0)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_logits = np.array(all_logits)
    all_gate_weights = np.array(all_gate_weights)

    # 1. Classification Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

    report_lines = []
    report_lines.append("=== Classification Metrics ===")
    report_lines.append(f"Accuracy: {acc:.4f}")
    report_lines.append(f"Macro Precision: {macro_precision:.4f}")
    report_lines.append(f"Macro Recall: {macro_recall:.4f}")
    report_lines.append(f"Macro F1: {macro_f1:.4f}\n")
    report_lines.append("Per-class Metrics:")
    for c in range(4):
        report_lines.append(f"Class {c}: Precision={precision[c]:.4f}, Recall={recall[c]:.4f}, F1={f1[c]:.4f}")

    # 5. Localisation error
    mean_loc_err = np.mean(loc_errors) if loc_errors else 0.0
    report_lines.append(f"\n=== Localisation Error ===")
    report_lines.append(f"Mean Euclidean Distance (Grid Units): {mean_loc_err:.4f}")

    # 6. Latency Benchmark
    report_lines.append(f"\n=== Latency Benchmark ===")
    model_cpu = model.to('cpu')
    dummy_mag = torch.randn(1, 1, 5, 5)
    dummy_thm = torch.randn(1, 1, 8, 8)
    dummy_gas = torch.randn(1, 20, 2)

    latencies = []
    with torch.no_grad():
        for _ in range(100): # Warmup
            model_cpu(dummy_mag, dummy_thm, dummy_gas)
        for _ in range(1000):
            start_t = time.time()
            model_cpu(dummy_mag, dummy_thm, dummy_gas)
            latencies.append((time.time() - start_t) * 1000)

    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    report_lines.append(f"Mean Latency (CPU): {mean_lat:.2f} ms ± {std_lat:.2f} ms")

    with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w') as f:
        f.write('\n'.join(report_lines))

    print('\n'.join(report_lines))

    # 2. Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # 3. ROC Curves
    plt.figure(figsize=(10, 8))
    # simple softmax
    probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
    for c in range(4):
        y_true = (all_labels == c).astype(int)
        y_score = probs[:, c]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {c} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    plt.close()

    # 4. Gate weights plot
    gate_means = []
    for c in range(4):
        mask = (all_labels == c)
        if mask.sum() > 0:
            c_weights = all_gate_weights[mask]
            gate_means.append(c_weights.mean(axis=0))
        else:
            gate_means.append(np.zeros(3))

    gate_means = np.array(gate_means)

    plt.figure(figsize=(10, 6))
    width = 0.25
    x = np.arange(4)
    plt.bar(x - width, gate_means[:, 0], width, label='Mag')
    plt.bar(x, gate_means[:, 1], width, label='Thermal')
    plt.bar(x + width, gate_means[:, 2], width, label='Gas')
    plt.xticks(x, [f'Class {c}' for c in range(4)])
    plt.ylabel('Average Gate Weight')
    plt.title('Gate Weights by Class')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'gate_weights_by_class.png'))
    plt.close()
