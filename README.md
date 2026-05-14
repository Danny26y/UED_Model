# Underground Explosive Device (UED) Detection Rover ML Pipeline

## 1. Project Title & Abstract
This project implements a multi-modal deep learning pipeline designed to enable an autonomous rover to detect Underground Explosive Devices (UEDs). The system fuses data from three distinct sensor modalities: a Magnetometer array detecting magnetic anomalies, a Thermal array identifying subsurface temperature variations, and Gas sensors monitoring airborne explosive/volatile compounds. By combining these signals, the model can safely and accurately distinguish between safe background environments and three distinct threat classes: Metallic UEDs, Plastic UEDs, and Improvised Explosive Devices (IEDs) with Ammonium Nitrate.

## 2. Architecture Overview
The core model is the **Hierarchical Multi-Level Fusion Network (HMLFNet)**, structured around a three-level hierarchy tailored for resource-constrained edge execution (e.g., Raspberry Pi 5).

- **Level 1 (Per-Modality Encoders):** Independent networks compress raw sensor inputs into dense 64-dimensional feature vectors. The Spatial sensors (Mag, Thermal) use CNNs with adaptive pooling, while the temporal Gas sensor uses a Bidirectional LSTM.
- **Level 2 (Adaptive Gated Fusion):** A Modality-Aware Gated Exchange module evaluates the confidence of each modality feature independently via an MLP, applying a softmax weighting to generate a unified fused feature alongside a residual cross-modal concatenate block.
- **Level 3 (Multitask Decision Heads):** Branches off into Classification, Localisation (bounding box regression), and Confidence scoring heads.

```
       [Mag 5x5]     [Thermal 8x8]     [Gas 20x2]
           |               |               |
     +-----+-----+   +-----+-----+   +-----+-----+
L1   |MagEncoder |   |TherLEncoder|   |GasEncoder |
     +-----+-----+   +-----+-----+   +-----+-----+
           |               |               |
           +-------+       |       +-------+
                   |       |       |
                 +-------------------+
L2               |        GFM        | (Adaptive Gated Fusion)
                 +-------------------+
                           |
           +---------------+---------------+
           |               |               |
L3   +-----------+   +-------------+   +--------------+
     |Class Head |   |Localise Head|   |Confidence Head|
     +-----------+   +-------------+   +--------------+
```

## 3. Sensor Specifications
| Sensor Modality | Configuration | Detection Target | Output Shape |
|-----------------|---------------|------------------|--------------|
| Magnetometer | 5x5 Grid | Magnetic field anomalies (µT) | (5, 5) |
| Thermal Array | 8x8 Grid (AMG8833) | Surface temperature variations (°C) | (8, 8) |
| Gas / Air Quality | MQ-2 & MQ-135 | Flammable gases, explosives (Rs/Ro) | (20, 2) |

## 4. Dataset
The dataset consists of 20,000 synthetic samples generated to mimic realistic environmental dynamics, structured dynamically at runtime. It's stratified by class:
- **Class 0 (Safe):** Background noise, ambient temperature, baseline gas levels.
- **Class 1 (Metallic UED):** Strong dipole magnetic signature, cool thermal heat sink anomaly, zero gas off-gassing.
- **Class 2 (Plastic UED):** Very weak magnetic signature, warm thermal signature (plastic insulator), elevated MQ-135 readings.
- **Class 3 (IED / AN):** Almost zero magnetic dipole, irregular thermal blob, aggressive dual MQ-2 and MQ-135 descent indicating strong gas plume.

## 5. Installation
```bash
git clone <repo>
cd ued_rover
pip install -r requirements.txt
# For Raspberry Pi 5 (CPU-only torch):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 6. Quick Start
Execute the entire pipeline end-to-end to generate the data, train the model, evaluate the accuracy, and launch visualisations:
```bash
python main.py pipeline --seed 42
```

## 7. Individual Commands
- `python main.py generate --output_dir ./data --seed 42`
  *Generates the synthetic `.npz` files and normalization stats.*
- `python main.py train --data_dir ./data --output_dir ./results --epochs 60`
  *Trains HMLFNet, executing early stopping, AMP, and learning rate scheduling.*
- `python main.py evaluate --data_dir ./data --model_path ./results/best_model.pt`
  *Calculates macro F1, localization error, confusion matrices, and ROC curves.*
- `python main.py export --model_path ./results/best_model.pt --output_dir ./model_export`
  *Exports the PyTorch state dict to an ONNX graph and a dynamic INT8 Quantised `.pt` file for edge deployment.*
- `python main.py visualise --data_dir ./data --model_path ./model_export/hmlf_net.pt --sample_idx 42`
  *Runs the interactive Matplotlib / Plotly visualization dashboard for a specific sample.*
- `python main.py visualise --data_dir ./data --model_path ./model_export/hmlf_net.pt --random`
  *Runs the dashboard on a randomly selected test sample.*
- `python main.py animate --data_dir ./data --model_path ./model_export/hmlf_net.pt`
  *Creates a rolling GIF simulating live inference parsing across 30 batch samples.*

## 8. Expected Performance
| Metric | Target | Notes |
|---|---|---|
| Test Accuracy | >95% | 4-class synthetic dataset |
| Macro F1 | >0.94 | Balanced across all classes |
| Inference Latency (CPU) | <100 ms | On Raspberry Pi 5 |
| Model Size (quantised) | <50 MB | INT8 dynamic quantisation |

## 9. Deployment on Raspberry Pi 5
1. Copy the exported artifacts `model_export/hmlf_net_quantised.pt` and `./data/dataset_stats.json` to the target Pi.
2. Ensure you install the optimized PyTorch CPU build linked in the Installation guide.
3. Configure the kernel to utilize `PREEMPT_RT` ensuring minimal execution interrupts. Use `isolcpus=2,3` to isolate inference workload cores.
4. Execute real-time inferences matching the structure parsed inside `visualise.py`: Apply identical Z-score normalizations, unsqueeze dummy batch axes, and call `model(mag, thermal, gas)`.

## 10. Real Sensor Integration Notes
When transitioning away from the synthetic generator:
- **Magnetometer:** Requires multiplexing 25 HMC5883L/QMC5883L sensors via an I2C multiplexer to populate the `(5, 5)` tensor.
- **Thermal:** The AMG8833 natively produces an 8x8 pixel map over I2C, mapping directly to `(8, 8)`.
- **Gas:** The MQ-2/MQ-135 sensors pass analog output through an MCP3008 ADC via SPI. Form a rolling temporal window over T=20 frames.

## 11. Limitations & Future Work
This pipeline presently utilizes purely synthetic environmental variables. While Gaussian baseline noise, correlated spatial noise, AR(1) temporal drift, and sensor clipping accurately mimic geological backgrounds, genuine field testing is mandatory. Future work will entail setting up an active learning pipeline allowing real sensor collections parsed during dummy field runs to progressively fine-tune the encoded PyTorch weights.

## 12. References
Refer to the foundational design documentation integrating Multi-Level Fusion, Bidirectional RNN configurations for chemical sniffer arrays, and spatial pooling hierarchies applied in embedded edge-compute platforms.
