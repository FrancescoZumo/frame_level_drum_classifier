import torch
import numpy as np
import librosa
import soundfile as sf
import argparse
from scipy.signal import chirp
from utils.dataset_preparation import SR, N_FFT, HOP_LENGTH, N_MELS, TARGET_CLASSES
from utils.model import DrumCNN
import os
from train import CHECKPOINTS_FOLDER

INFERENCE_FOLDER = 'inference_files'

# ---- Config (must match training) ----

# Sonification: each class gets a distinct sine tone frequency
CLASS_FREQS = {
    'kick':  200,    # low thump
    'snare': 400,   # mid crack
    'hihat': 800,  # high tick
}
TONE_DURATION = 0.05   # seconds, length of each beep
TONE_AMPLITUDE = 0.6


def load_model(checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    context = checkpoint['context']
    n_mels  = checkpoint['n_mels']

    model = DrumCNN(n_mels=n_mels, context=context, n_classes=3).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, context


def extract_features(audio_path):
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT,
                                          hop_length=HOP_LENGTH, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    delta   = librosa.feature.delta(log_mel)
    delta2  = librosa.feature.delta(log_mel, order=2)
    features = np.stack([log_mel, delta, delta2], axis=0)  # (3, 80, T)
    duration = len(y) / SR
    return features, duration, y


def run_inference(model, features, context=3, batch_size=4096, device='cpu', threshold=0.5):
    """Run batched inference over all frames. Returns (n_frames, 3) binary predictions."""
    # features: (3, 80, T) → (T, 3, 80)
    features = features.transpose(2, 0, 1)
    T = len(features)

    pad = np.zeros((context, 3, N_MELS), dtype=np.float32)
    features_padded = np.concatenate([pad, features, pad], axis=0)

    # build all windows: (T, 2*context+1, 3, 80) → (T, 3, 2*context+1, 80)
    windows = np.stack([features_padded[i:i + 2*context + 1]
                        for i in range(T)], axis=0)          # (T, window, 3, 80)
    windows = windows.transpose(0, 2, 1, 3)                  # (T, 3, window, 80)

    all_preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, T, batch_size):
            batch = torch.tensor(windows[i:i+batch_size], dtype=torch.float32).to(device)
            logits = model(batch)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)

    probs = np.concatenate(all_preds, axis=0)       # (T, 3)
    preds = (probs > threshold).astype(np.float32)  # (T, 3) binary
    return preds, probs


def preds_to_onsets(preds):
    """
    Convert frame-level binary predictions to onset times in seconds.
    An onset is a rising edge: frame goes from 0 to 1.
    Returns dict: {class_name: [onset_time_seconds, ...]}
    """
    onsets = {cls: [] for cls in TARGET_CLASSES}
    for cls_idx, cls in enumerate(TARGET_CLASSES):
        cls_preds = preds[:, cls_idx]
        for i in range(1, len(cls_preds)):
            if cls_preds[i] == 1 and cls_preds[i-1] == 0:
                onset_time = i * HOP_LENGTH / SR
                onsets[cls].append(onset_time)
    return onsets


def generate_tone(freq, duration, sr):
    """Generate a pure sine tone."""
    tone = librosa.tone(freq, duration=duration, sr=sr)
    return tone


def sonify_onsets(onsets, total_duration, sr=SR):
    """
    Place a tone at each onset time.
    Each class gets a different frequency.
    Returns stereo audio: original left, sonification right.
    """
    tone_samples = int(TONE_DURATION * sr)

    for cls, onset_times in onsets.items():
        output = np.zeros(int(total_duration * sr), dtype=np.float32)
        freq = CLASS_FREQS[cls]
        tone = generate_tone(freq, TONE_DURATION, sr)
        for t in onset_times:
            start = int(t * sr)
            end   = min(start + tone_samples, len(output))
            output[start:end] += tone[:end - start]

        # normalize to avoid clipping
        max_val = np.abs(output).max()
        if max_val > 0:
            output = output / max_val * 0.9

        sf.write(os.path.join(INFERENCE_FOLDER, "{}.wav".format(cls)), output, sr)

    return output


def transcribe(audio_path, checkpoint_path, output_path, threshold=0.5, device='cpu'):
    # I want to test inference only on cpu for realistic processing times
    print(f"Using device: {device}")

    print("Loading model...")
    model, context = load_model(checkpoint_path, device)


    from torchinfo import summary

    summary(model, (4096, 3, 7, 96))
    exit(0)

    print("Extracting features...")
    features, duration, original_audio = extract_features(audio_path)

    print("Running inference...")
    preds, probs = run_inference(model, features, context=context,
                                 device=device, threshold=threshold)

    onsets = preds_to_onsets(preds)
    for cls, times in onsets.items():
        print(f"  {cls}: {len(times)} onsets detected")

    print("Generating sonification for each class...")
    sonify_onsets(onsets, duration, sr=SR)

    print("Done")



if __name__ == "__main__":
    import time
    start = time.time()
    parser = argparse.ArgumentParser(description="Drum transcription inference")
    parser.add_argument("audio", nargs='?', type=str, help="Path to input audio file", default=os.path.join(INFERENCE_FOLDER, "HighwaytoHell.mp3"))
    parser.add_argument("checkpoint", nargs='?', type=str, help="Path to model checkpoint (.pth)", default=os.path.join(CHECKPOINTS_FOLDER, "drum_cnn_high_recall.pth"))
    parser.add_argument("output", nargs='?', type=str, help="Path to output audio file (.wav)", default=os.path.join(CHECKPOINTS_FOLDER, "test.wav"))
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    args = parser.parse_args()



    transcribe(args.audio, args.checkpoint, args.output, threshold=args.threshold)

    print(f"Execution completed in {time.time() - start:.2f} seconds")