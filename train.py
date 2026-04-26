import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import numpy as np
from utils.model import DrumCNN
from utils.dataset_preparation import prepare_for_cnn, load_training_data, SR, N_FFT, HOP_LENGTH, CACHE_PATH
import os
from utils.dataloader import DrumDataset

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
          n_epochs=100, lr=1e-3, device='cpu', patience=10):

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
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)
    return model


# def main():
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Using device: {device}")

#     # --- sanity check: overfit on 3 tracks ---
#     print("=== Sanity check: overfitting on 3 tracks ===")
#     tracks = get_train_data(max_elements=3)
#     X, y = prepare_for_cnn(tracks, context=3)
#     print(f"X: {X.shape}, y: {y.shape}")
#     print(f"Class balance — kick: {y[:,0].mean():.3f}, "
#           f"snare: {y[:,1].mean():.3f}, hihat: {y[:,2].mean():.3f}")

#     # use same data for train and val — we just want to see if it can overfit
#     model = DrumCNN(n_mels=80, context=3, n_classes=3).to(device)
#     model = train(model, X, y, X, y,
#                   n_epochs=30, batch_size=512, lr=1e-3, 
#                   device=device, patience=30)  # high patience to not stop early

#     # if F1 scores reach >0.9 on train==val, pipeline is correct
#     # then move to full training with proper train/val split

def main():
    # params
    window_context = 3
    n_workers = os.cpu_count() - 1  # leave one core free for the OS
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- full training ---
    print("1) extracting features and labels from training data")

    tracks = load_training_data(max_elements=500, n_workers=n_workers, load_if_available=True)
    
    print("2)  performing train test split")
    np.random.seed(42)
    # split at track level, so I have no leakage
    indices = np.random.permutation(len(tracks))
    # 80/10/10 train val test ratio
    n_train = int(0.8 * len(tracks))
    n_val   = int(0.1 * len(tracks))

    train_tracks = [tracks[i] for i in indices[:n_train]]
    val_tracks   = [tracks[i] for i in indices[n_train:n_train + n_val]]
    test_tracks  = [tracks[i] for i in indices[n_train + n_val:]]


    print("3) building datasets and dataloaders")
    n_dataloader_workers = 0 if os.name == 'nt' else n_workers
    train_dataset = DrumDataset(train_tracks, context=window_context)
    val_dataset   = DrumDataset(val_tracks,   context=window_context)
    test_dataset  = DrumDataset(test_tracks,  context=window_context)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True,  num_workers=n_dataloader_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=1024, shuffle=False, num_workers=n_dataloader_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=1024, shuffle=False, num_workers=n_dataloader_workers)

    # compute pos weights from train labels only
    print("4) computing class weights")
    y_train_all = np.concatenate([labels for _, labels in train_tracks], axis=0)
    pos_weight = compute_pos_weights(y_train_all).to(device)
    print(f"Class balance — kick: {y_train_all[:,0].mean():.3f}, "
          f"snare: {y_train_all[:,1].mean():.3f}, hihat: {y_train_all[:,2].mean():.3f}")

    print("5) training")
    model = DrumCNN(n_mels=80, context=window_context, n_classes=3).to(device)
    model = train(model, train_loader, val_loader, pos_weight,
                  n_epochs=100, lr=1e-4, device=device, patience=10)

    print("\n=== Test set evaluation ===")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    test_loss, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f} | "
          f"F1 kick: {test_f1[0]:.3f} snare: {test_f1[1]:.3f} hihat: {test_f1[2]:.3f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'context': window_context,
        'n_mels': 80,
        'n_classes': 3,
        'sr': SR,
        'hop_length': HOP_LENGTH,
        'n_fft': N_FFT,
    }, os.path.join(CHECKPOINTS_FOLDER, 'drum_cnn_{}.pth'.format(test_loss)))
    print("Model saved to drum_cnn.pth")

if __name__ == "__main__":
    main()