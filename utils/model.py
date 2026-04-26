import torch
import torch.nn as nn

class DrumCNN(nn.Module):
    def __init__(self, n_mels=80, context=3, n_classes=3):
        super().__init__()

        #TODO for now n_mels not used, but expected to be 80
        assert n_mels == 80, "TODO"
        
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # pool only frequency axis → (16, 7, 40)

            # Block 2
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # → (32, 7, 20)
        )

        # context=3 → window=7 frames, after pooling freq: 80→40→20
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * (2 * context + 1) * 20, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
            # no sigmoid here — BCEWithLogitsLoss expects raw logits
        )

    def forward(self, x):
        # x: (batch, 3, 7, 80)
        x = self.cnn(x)
        return self.fc(x)  # (batch, 3)