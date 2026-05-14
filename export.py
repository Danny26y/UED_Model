import torch
import os
import onnx
import onnxruntime as ort

def export_model(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    model.to('cpu')

    # Save unquantised PyTorch model dict required for inference
    torch.save(model.state_dict(), os.path.join(output_dir, 'hmlf_net.pt'))

    dummy_mag = torch.randn(1, 1, 5, 5)
    dummy_thm = torch.randn(1, 1, 8, 8)
    dummy_gas = torch.randn(1, 20, 2)

    onnx_path = os.path.join(output_dir, 'hmlf_net.onnx')

    torch.onnx.export(
        model,
        (dummy_mag, dummy_thm, dummy_gas),
        onnx_path,
        input_names=['mag', 'thermal', 'gas'],
        output_names=['logits', 'loc', 'conf', 'gate_weights'],
        dynamic_axes={
            'mag': {0: 'batch_size'},
            'thermal': {0: 'batch_size'},
            'gas': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'loc': {0: 'batch_size'},
            'conf': {0: 'batch_size'},
            'gate_weights': {0: 'batch_size'}
        }
    )

    # INT8 Quantisation
    quantised_path = os.path.join(output_dir, 'hmlf_net_quantised.pt')
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8
    )
    torch.save(quantized_model.state_dict(), quantised_path)

    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    quant_size = os.path.getsize(quantised_path) / (1024 * 1024)

    print("\n=== Export Sizes ===")
    print(f"ONNX Model: {onnx_size:.2f} MB")
    print(f"Quantised PT Model: {quant_size:.2f} MB")

    # Verify ONNX
    ort_session = ort.InferenceSession(onnx_path)
    inputs = {
        'mag': dummy_mag.numpy(),
        'thermal': dummy_thm.numpy(),
        'gas': dummy_gas.numpy()
    }
    outputs = ort_session.run(None, inputs)

    print("\n=== ONNX Verification ===")
    print(f"Logits shape: {outputs[0].shape}")
    print(f"Loc shape: {outputs[1].shape}")
    print(f"Conf shape: {outputs[2].shape}")
    print(f"Gate shape: {outputs[3].shape}")
