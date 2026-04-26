from torch.utils.data import Dataset
import torch
import numpy as np
class DrumDataset(Dataset):
    def __init__(self, tracks, context=7):
        self.context = context
        self.index = []  # list of (track_idx, frame_idx)

        # build an index of all valid (track, frame) pairs
        for track_idx, (features, labels) in enumerate(tracks):
            n_frames = labels.shape[0]
            for frame_idx in range(n_frames):
                self.index.append((track_idx, frame_idx))

        self.tracks = tracks

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        track_idx, frame_idx = self.index[idx]
        features, labels = self.tracks[track_idx]

        # features: (3, 80, T)
        T = features.shape[-1]
        context = self.context

        # pad if needed at boundaries
        start = frame_idx - context
        end = frame_idx + context + 1

        pad_left  = max(0, -start)
        pad_right = max(0, end - T)

        start = max(0, start)
        end   = min(T, end)

        window = features[:, :, start:end]  # (3, 80, actual_width)

        # pad with zeros if at boundary
        if pad_left > 0 or pad_right > 0:
            window = np.pad(window, ((0,0), (0,0), (pad_left, pad_right)))

        # window: (3, 80, 2*context+1) → (3, 2*context+1, 80)
        window = window.transpose(0, 2, 1)

        return (torch.tensor(window, dtype=torch.float32),
                torch.tensor(labels[frame_idx], dtype=torch.float32))