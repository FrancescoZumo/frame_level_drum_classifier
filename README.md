# Frame Level Drum Classifier
The goal of this project is to transcribe the kicks, snares and hihats from any input audio, with enough temporal precision and low computational cost.

## Method Overview

I chose to stick to a simple architecture, a 4D CNN that receives mel spectrograms as input feature and outputs probabilities used to predict the presence or absence of each class at every frame.

The temporal resolution of this approach was set to 11.6ms (i.e. STFT with 256 hop size on 22050HZ audio), to ensure precise onset times. 

The training was performed on a dataset with the following specifications:
- 614 full song mixes, coupled with ground truth, frame level annotations
- 614 drum only songs, coupled with ground truth, frame level annotations
Each song in the first group has its correspondent in the second one.

The combination of the two was used to train the CNN.

Class level F1 Score was the main evaluation metric on the test set.

## Training Dataset
 - STAR Dataset, https://zenodo.org/records/15690078.
Among all analyzed options, this dataset provides drums annotations for both drums only recordings and complete song mix.
Also, the annotation schema is relatively simple to parse, with respect to other datasets (e.g. IDMT-SMT-Drums https://www.idmt.fraunhofer.de/en/publications/datasets/drums.html)

## Feature Extraction
- I chose mel-spectrograms as main feature, as long as their 1st and 2nd derivative computed across time axis.
Together they are fed as a 3-channel feature to the CNN.

- I used the following parameters for computing STFT and mel spectrograms:
```
SR = 22050         # captures the necessary frequency range for drums, though certain papers even resample down to 16KHz
HOP_LENGTH = 256   # determines ~11.6 ms time resolution
N_MELS = 96        # usually 80 or 96 are standard choices
N_FFT = 1024       # it determines ~46 ms time resolution and ~21Hz freq resolution
```

## Model architecture
I chose to implement a simple, lightweight CNN, starting from what was used in paper https://ismir2025program.ismir.net/poster_130.html 

The CNN comprises 3 convolutional blocks and 2 dense layers.
It receives 4D inputs with dimensions (BATCH, CHANNELS, CONTEXT_WINDOW, MELS)

- with respect to reference paper, I reduced the the number of convolutional blocks from 5 to 3, and the context window from 25 to 11, to reduce model size.

The model's number of params is 1,105,667.

## Evaluation results
TODO F1 Score on test set
TODO Per-class precision and recall on test set

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

## Possible Improvements
- Dataset audio samples belong to multiple classes (Drum kit, playing style). I should balance their presence equally in train val and test sets.
- Instead of fixed 0.5 class prediction threshold, each class could have a custom threshold that maximizes F1 score on test set.
- For evaluation, F1 score could be computed with a tolerance window across time (e.g. a kick predicted slightly earlier or later is still considered as True Positive)
- Optimize model training to reduce RAM usage (right now all dataset constantly stays in RAM)
