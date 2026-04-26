from torch.utils.data import Dataset
import torch
import numpy as np
class DrumsDataset(Dataset):
    def __init__(self, tracks, context=7):
        self.context = context
        self.tracks = tracks  # keep as numpy — no bulk conversion

        # vectorized index building
        track_indices, frame_indices = [], []
        for track_idx, (features, labels) in enumerate(tracks):
            n_frames = labels.shape[0]
            track_indices.append(np.full(n_frames, track_idx, dtype=np.int32))
            frame_indices.append(np.arange(n_frames,            dtype=np.int32))

        self.track_indices = np.concatenate(track_indices)
        self.frame_indices = np.concatenate(frame_indices)

    def __len__(self):
        return len(self.track_indices)  # fixed bug

    def __getitem__(self, idx):
        track_idx = self.track_indices[idx]
        frame_idx = self.frame_indices[idx]
        features, labels = self.tracks[track_idx]

        T = features.shape[-1]
        context = self.context

        start = frame_idx - context
        end   = frame_idx + context + 1

        pad_left  = max(0, -start)
        pad_right = max(0, end - T)

        start = max(0, start)
        end   = min(T, end)

        window = features[:, :, start:end].copy()  # (3, 80, window)

        if pad_left > 0 or pad_right > 0:
            window = np.pad(window, ((0, 0), (0, 0), (pad_left, pad_right)))

        # (3, 80, 2*context+1) → (3, 2*context+1, 80)
        window = window.transpose(0, 2, 1)

        return (torch.from_numpy(window).float(),
                torch.from_numpy(labels[frame_idx].copy()).float())



    
    
