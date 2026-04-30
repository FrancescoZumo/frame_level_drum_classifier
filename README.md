# Frame Level Drum Classifier
The goal of this project is to transcribe the kicks, snares and hihats from any input audio, with enough temporal precision and low computational cost.

## Method Overview

I trained a 4D CNN that receives mel spectrograms as input feature and outputs probabilities used to predict the presence or absence of each class at every frame.

The temporal resolution of this approach was set to 11.7ms (i.e. STFT with 256 hop size on 22050Hz audio). 

The training was performed on a dataset of audio recordings with drums class-level annotations, with temporal resolution of ~10ms:

This was modeled as a binary classification problem with three independent predictions at each frame (kick, snare, hi-hat), since the three hits can occour simultaneously.



## Training Dataset
 - STAR Dataset, https://zenodo.org/records/15690078.
Among all analyzed options, this dataset provides drums annotations for both drums only recordings and complete song mix. It was interesting for making experiments with multiple approaches.
Also, the annotation schema is relatively simple to parse, with respect to other datasets.

For the model training, a subset of this datased was used and it's composed as follows:
- 614 full song mixes, coupled with ground truth, frame level annotations
- 614 drums only songs, coupled with ground truth, frame level annotations
Each song in the first group has its correspondent in the second one.

The combination of the two was used to train the CNN.

### Dataset Annotations
The dataset provides annotations for 18 classes. I foolowed the mapping proposed by the authors to reduce them to 3.
```
# Map STAR classes to only Three classes: Kick, Snare, Hi-hat
# see https://transactions.ismir.net/articles/244/files/6888ab991b2f2.pdf page 255 for reference to names mapping
CLASS_MAP = {
    'BD': 'kick',    # Bass Drum
    'SD': 'snare',   # Snare Drum
    'CHH': 'hihat',   # Closed Hi-Hat
    'PHH': 'hihat',   # Pedal Hi-Hat
    'OHH': 'hihat',   # Open Hi-Hat
    # LT, MT, HT, CY etc. → ignored (toms, cymbals)
}
```

## Feature Extraction
- I chose mel-spectrograms as main feature, as well as their 1st and 2nd derivative computed across time axis.
Together they are fed as a 3-channel feature to the CNN.

- I used the following parameters for computing STFT and mel spectrograms:
```
SR = 22050         # captures the necessary frequency range for drums, though certain papers even resample down to 16KHz
HOP_LENGTH = 256   # determines ~11.7 ms time resolution
N_MELS = 96        # usually 80 or 96 are standard choices
N_FFT = 1024       # it determines ~46 ms time resolution and ~21Hz freq resolution
```

## Model architecture
I chose to implement a lightweight CNN, starting from what was used in paper https://ismir2025program.ismir.net/poster_130.html 

The CNN comprises 3 convolutional blocks and 2 dense layers.
It receives 4D inputs with dimensions (BATCH, CHANNELS, CONTEXT_WINDOW, MELS)

- with respect to reference paper, I reduced the the number of convolutional blocks from 5 to 3, and the context window from 25 to 7, to reduce model size.

- The model's number of params is 712,451.

## Evaluation results

P: Precision

R: Recall

F1: F1-Score

### Results
```

Early stopping at epoch:
Epoch 026 | train_loss: 0.0517 | val_loss: 0.0668 | F1 kick: 0.898 snare: 0.841 hihat: 0.821


=== Test set evaluation ===
Test loss: 0.0605 | F1 kick: 0.905 snare: 0.847 hihat: 0.712

=== Per-class evaluation ===
  kick   — P: 0.924  R: 0.886  F1: 0.905
  snare  — P: 0.856  R: 0.838  F1: 0.847
  hihat  — P: 0.771  R: 0.662  F1: 0.712
```

#### Discussion
Kick predictions show the best performance among three classes, while hihats are the least precise class.
One reason may be that other drum instruments overlap in frequency domain with hihats (i.e. cymbals) and their onset and timbre are relatively similar.
Additionally, the kick's frequency band is the most separable from other instruments, hence this for sure contributed to this result.
Finally, hihats in particular would benefit by providing a positional weight (slightly greater than 1) to their class while computing the binary cross entropy loss,
so that the recall could increase. 

## Setup instructions
### Install requirements
```
conda create --name drums_classifier --file reqirements.txt
```

### run the web UI demo to try the model exported to onnx
```
cd webUI
uvicorn server:app --reload
```

### train the CNN locally
- setup cuda for your device and adjust batch size to your VRAM (current setup works for 4GB)
- setup dataset max size according to your RAM (current setup works for 16GB)

- Download the STAR Dataset From zenodo: https://zenodo.org/records/15690078 and setup paths inside python scripts
- Adjust Number of tracks to load from dataset
- run train script
```
python train.py
```

- find the trained model in 'checkpoints' folder

### export the trained model to ONNX
- run export_to_onnx.py
```
python export_to_onnx.py [path_to_checkpoint_file.pth]
```

## Previous Experiments

### Experiment 1: 
```
=== Test set evaluation ===
Test loss: 0.2286 | F1 kick: 0.835 snare: 0.648 hihat: 0.627

=== Per-class evaluation ===
  kick   — P: 0.734  R: 0.968  F1: 0.835
  snare  — P: 0.486  R: 0.971  F1: 0.648
  hihat  — P: 0.483  R: 0.896  F1: 0.627
```

#### Discussion
Although Kick predictions are acceptable, there is an extreme tendency to recall. 
The CNN during training learned to prefer false positives to false negatives. 
In the next experiment I will try to reduce or remove the positional weights computed for the Binary Cross Entropy Loss.

### Experiment 2: 
```
=== Test set evaluation ===
Test loss: 0.1209 | F1 kick: 0.882 snare: 0.794 hihat: 0.694

=== Per-class evaluation ===
  kick   — P: 0.831  R: 0.938  F1: 0.882
  snare  — P: 0.697  R: 0.923  F1: 0.794
  hihat  — P: 0.605  R: 0.815  F1: 0.694
```
#### Discussion
In this experiment a square root was applied to the original positional weights of each class. 
This result demonstrated their negative influence on this task, hence for the final version they will be removed.

## Possible Improvements
- Dataset audio samples belong to multiple classes (Drum kit, playing style). I should balance their presence equally in train val and test sets.
- Instead of fixed 0.5 class prediction threshold, each class could have a custom threshold that maximizes F1 score on test set.
- For evaluation, F1 score could be computed with a tolerance window across time (e.g. a kick predicted slightly earlier or later is still considered as True Positive)
- Optimize model training to reduce RAM usage (right now all dataset constantly stays in RAM)

## AI Usage
- NotebookLM was used to retrieve and organize information from research papers to guide preliminary design choices
- Claude (via Web) was used as a programming assistant for building the training pipeline and for reviewing design choices.
- The webUI demo application has been mostly written by Claude. 

*All AI-generated code and suggestions were reviewed and validated by the author.*
