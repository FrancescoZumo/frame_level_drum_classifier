import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import numpy as np
from utils.model import DrumCNN
from utils.dataset_preparation import load_training_data, SR, N_FFT, HOP_LENGTH, N_MELS, TARGET_CLASSES
import os
from utils.DrumsDataset import DrumsDataset
from sklearn.metrics import precision_score, recall_score, f1_score

CHECKPOINTS_FOLDER = 'checkpoints'

def compute_pos_weights(y_train):
    """Compute positive weights for BCEWithLogitsLoss to handle class imbalance."""
    n_neg = (y_train == 0).sum(axis=0)
    n_pos = (y_train == 1).sum(axis=0)
    pos_weight = n_neg / np.maximum(n_pos, 1)  # avoid division by zero
    print(f"Pos weights — kick: {pos_weight[0]:.1f}, snare: {pos_weight[1]:.1f}, hihat: {pos_weight[2]:.1f}")
    return torch.tensor(pos_weight, dtype=torch.float32)


def evaluate(model: nn.Module, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > threshold).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y_batch.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    avg_loss = total_loss / len(loader)

    return avg_loss, f1_per_class


def train(model, train_loader: DataLoader, val_loader: DataLoader, pos_weight,
          n_epochs=100, lr=1e-3, device='cuda', patience=10, experiment_name: str = 'experiment'):

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = np.inf
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(n_epochs):
        # --- train ---
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- validate ---
        val_loss, f1 = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1:03d} | "
              f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
              f"F1 kick: {f1[0]:.3f} snare: {f1[1]:.3f} hihat: {f1[2]:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict().copy()
            # save checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'context': train_loader.dataset.get_window_context(),
                'n_mels': N_MELS,
                'n_classes': len(TARGET_CLASSES),
                'sr': SR,
                'hop_length': HOP_LENGTH,
                'n_fft': N_FFT,
            }, os.path.join(CHECKPOINTS_FOLDER, 'epoch_{}_{}.pth'.format(epoch, experiment_name)))
            print("Model saved to {}".format( os.path.join(CHECKPOINTS_FOLDER, 'drum_cnn_epoch_{}.pth'.format(epoch))))

        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)
    return model


def evaluate_detailed(model: torch.nn.Module, loader, device, thresholds=None):
    if thresholds is None:
        thresholds = np.array([0.5, 0.5, 0.5]) # for now fixed

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(device))
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds  = (probs > thresholds[np.newaxis, :]).astype(int)
            all_preds.append(preds)
            all_labels.append(y_batch.numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print("\n=== Per-class evaluation ===")
    for cls_idx, cls in enumerate(TARGET_CLASSES):
        p  = precision_score(all_labels[:, cls_idx], all_preds[:, cls_idx], zero_division=0)
        r  = recall_score(   all_labels[:, cls_idx], all_preds[:, cls_idx], zero_division=0)
        f1 = f1_score(       all_labels[:, cls_idx], all_preds[:, cls_idx], zero_division=0)
        print(f"  {cls:6s} — P: {p:.3f}  R: {r:.3f}  F1: {f1:.3f}")


def main():
    # params
    window_context = 3
    n_workers = os.cpu_count() - 1  # leave one core free for the OS
    
    experiment_name = "final"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- full training ---
    print("1) extracting features and labels from training data")

    paired_tracks = load_training_data(max_elements=614, n_workers=n_workers, load_if_available=False)
    
    print("2)  performing train test split")

    np.random.seed(42)
    # split at track level, so I have no leakage
    indices = np.random.permutation(len(paired_tracks))
    # 80/10/10 train val test ratio
    n_train = int(0.8 * len(paired_tracks))
    n_val   = int(0.1 * len(paired_tracks))

    train_pairs = [paired_tracks[i] for i in indices[:n_train]]
    val_pairs   = [paired_tracks[i] for i in indices[n_train:n_train + n_val]]
    test_pairs  = [paired_tracks[i] for i in indices[n_train + n_val:]]

    # flatten pairs into track lists
    # train gets both mix and re-synthesized
    train_tracks = [track for pair in train_pairs for track in pair]

    # val and test get only mix since evaluation must reflect real use case
    val_tracks  = [pair[0] for pair in val_pairs]   # index 0 = mix
    test_tracks = [pair[0] for pair in test_pairs]  # index 0 = mix


    print("3) building datasets and dataloaders")
    n_dataloader_workers = 0 if os.name == 'nt' else n_workers
    train_dataset = DrumsDataset(train_tracks, context=window_context, augment=False)
    val_dataset   = DrumsDataset(val_tracks,   context=window_context)
    test_dataset  = DrumsDataset(test_tracks,  context=window_context)

    train_loader = DataLoader(train_dataset, batch_size=1024 * 4, shuffle=True,  num_workers=n_dataloader_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=1024 * 4, shuffle=False, num_workers=n_dataloader_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=1024 * 4, shuffle=False, num_workers=n_dataloader_workers)

    # compute pos weights from train labels only
    print("4) computing class weights")
    y_train_all = np.concatenate([labels for _, labels in train_tracks], axis=0)
    pos_weight = compute_pos_weights(y_train_all).to(device)

    print(f"Class balance — kick: {y_train_all[:,0].mean():.3f}, "
          f"snare: {y_train_all[:,1].mean():.3f}, hihat: {y_train_all[:,2].mean():.3f}")

    print("5) training")
    model = DrumCNN(n_mels=N_MELS, context=window_context, n_classes=3).to(device)
    model = train(model, train_loader, val_loader, pos_weight, n_epochs=100, 
                  lr=1e-3, device=device, patience=5, experiment_name=experiment_name)

    print("\n=== Test set evaluation ===")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    test_loss, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f} | "
          f"F1 kick: {test_f1[0]:.3f} snare: {test_f1[1]:.3f} hihat: {test_f1[2]:.3f}")
    # 
    evaluate_detailed(model, test_loader, device)

    torch.save({
        'model_state_dict': model.state_dict(),
        'context': window_context,
        'n_mels': N_MELS,
        'n_classes': len(TARGET_CLASSES),
        'sr': SR,
        'hop_length': HOP_LENGTH,
        'n_fft': N_FFT,
        'n_tracks': len(paired_tracks),
        'test_f1': test_f1,
    }, os.path.join(CHECKPOINTS_FOLDER, 'drum_cnn_{}.pth'.format(experiment_name)))
    print("Model saved to {}".format(os.path.join(CHECKPOINTS_FOLDER, 'drum_cnn_{}.pth'.format(experiment_name))))

if __name__ == "__main__":
    main()