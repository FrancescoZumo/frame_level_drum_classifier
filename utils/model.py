import torch.nn as nn

# simpler version of CNN used in paper https://ismir2025program.ismir.net/poster_130.html 

# here I want a lightweight version, so less conv blocks but same philosophy

# n_mels will be 80 or 96, I don't expect much difference

# context frames is low compared to paper (7 vs 25), maybe I could Increase it?

# Added more dropout since during training model overfits early, If Possible I would like to reduce it


# before adding the third conv block ,try adding more dropout layers

class DrumCNN(nn.Module):
    def __init__(self, n_mels=96, context=3, n_classes=3):
        super().__init__()

        assert n_mels % 8 == 0, f"n_mels must be divisible by 8, got {n_mels}"

        # 3 convolutional blocks 
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # pool only frequency axis → (16, 7, 48)
            nn.Dropout2d(0.2),
            
            # Block 2
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # → (32, 7, 24)
            nn.Dropout2d(0.2),
            
            # Block 3
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # → (64, 7, 12)
            nn.Dropout2d(0.2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (2 * context + 1) * (n_mels // 8), 128), # dynamically computed Dense layer size
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(128, n_classes)
            # no sigmoid here — BCEWithLogitsLoss expects raw logits
        )

    def forward(self, x):
        # x: (batch, 3, 7, 96)
        x = self.cnn(x)
        return self.fc(x)  # (batch, 3)


# old test version
# class DrumCNN(nn.Module):
#     def __init__(self, n_mels=80, context=3, n_classes=3):
#         super().__init__()

#         #TODO for now n_mels not used, but expected to be 80
#         assert n_mels == 80, "TODO"
        
#         self.cnn = nn.Sequential(
#             # Block 1
#             nn.Conv2d(3, 16, kernel_size=(3, 3), padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d((1, 2)),  # pool only frequency axis → (16, 7, 40)

#             # Block 2
#             nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d((1, 2)),  # → (32, 7, 20)
#         )

#         # context=3 → window=7 frames, after pooling freq: 80→40→20
#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(32 * (2 * context + 1) * 20, 64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, n_classes)
#             # no sigmoid here — BCEWithLogitsLoss expects raw logits
#         )

#     def forward(self, x):
#         # x: (batch, 3, 7, 80)
#         x = self.cnn(x)
#         return self.fc(x)  # (batch, 3)