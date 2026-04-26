import torch
import onnx
import os
from train import DrumCNN 

from train import CHECKPOINTS_FOLDER

def export_to_onnx(checkpoint_path: str, output_path: str = 'drum_cnn.onnx'):
    
    device = 'cpu'  # always export on CPU
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # read params from checkpoint
    context   = checkpoint['context']
    n_mels    = checkpoint['n_mels']
    n_classes = checkpoint['n_classes']
    
    print(f"Loaded checkpoint: context={context}, n_mels={n_mels}, n_classes={n_classes}")
    print(f"  trained on {checkpoint.get('n_tracks', '?')} tracks")
    print(f"  test F1: {checkpoint.get('test_f1', '?')}")

    # load model
    model = DrumCNN(n_mels=n_mels, context=context, n_classes=n_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # dummy input matching training shape: (batch, channels, window, mels)
    window_size = 2 * context + 1
    dummy_input = torch.randn(1, 3, window_size, n_mels)

    # export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['logits'],
        dynamic_axes={
            'input':  {0: 'batch_size'},
            'logits': {0: 'batch_size'},
        }
    )
    # force merge into single file after export
    import onnx
    from onnx.external_data_helper import load_external_data_for_model

    model_onnx = onnx.load(output_path)
    load_external_data_for_model(model_onnx, os.path.dirname(output_path))

    # save as single self-contained file
    onnx.save(model_onnx, output_path, save_as_external_data=False)

    os.remove(output_path + ".data")  # remove external data file no longer needed

    print("Saved as single file")
    print(f"Exported to {output_path}")

    # verify
    model_onnx = onnx.load(output_path)
    onnx.checker.check_model(model_onnx)
    print("ONNX model check passed")

    # sanity check with onnxruntime
    import onnxruntime as ort
    import numpy as np
    session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    dummy_np = dummy_input.numpy()
    logits_onnx  = session.run(None, {'input': dummy_np})[0]
    logits_torch = model(dummy_input).detach().numpy()
    max_diff = np.abs(logits_onnx - logits_torch).max()
    print(f"Max output difference PyTorch vs ONNX: {max_diff:.6f}")
    if max_diff < 1e-4:
        print("Outputs match")
    else:
        print("WARNING: outputs differ — check export")

    # print model size
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Model size: {size_mb:.2f} MB")

    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, nargs='?', help="Path to .pth checkpoint", \
                        default=os.path.join(CHECKPOINTS_FOLDER, "drum_cnn_1024_fft.pth"))
    parser.add_argument("--output", type=str, default=os.path.join("webUI", "drum_cnn.onnx"))
    args = parser.parse_args()

    export_to_onnx(args.checkpoint, args.output)