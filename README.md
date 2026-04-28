# Frame Level Drum Classifier
I trained a lightweight CNN to transcribe kicks, snares and hi-hats from any input audio file.

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

## Method Overview
The Goal is to transcribe the kicks, snares, hihats from any input audio, with enough temporal precision and low computational cost.

I chose to stick to the simplest architecture, a 4D CNN that receives 3-channel mel spectrograms as input feature (2nd and 3rd channels are respectively 1st and 2nd derivative of mel spectrogram) 

The CNN outputs 3 logits, that are subsequently converted to probabilities with sigmoid function and used to predict the presence or absence of each class at every frame.

## Training Dataset
 - STAR Dataset, https://zenodo.org/records/15690078.
Among all analyzed options, This dataset provides drums annotations for both drums only recordings and complete song mix.
Also, the annotation schema is relatively simple to parse, with respect to other datasets (e.g. IDMT-SMT-Drums https://www.idmt.fraunhofer.de/en/publications/datasets/drums.html)

## Model architecture
I chose to implement a simple, lightweight CNN, starting from what was used in paper https://ismir2025program.ismir.net/poster_130.html 

The CNN comprises 3 convolutional blocks and 2 dense layers.
It receives 4D inputs with dimensions (BATCH, CHANNELS, CONTEXT_WINDOW, MELS)


## Design decisions and trade-offs
### Feature Extraction
- I used mel-spectrograms as main feature, as long as their 1st and 2nd derivative computed across time axis.
- 

### CNN 
- with respect to reference paper, I reduced the the number of convolutional blocks from 5 to 3, and the context window from 25 to 11, to reduce model size.


## Evaluation results
TODO F1 Score on test set
TODO Per-class precision and recall on test set

## Possible Improvements
- Dataset audio samples belong to multiple classes (Drum kit, playing style). I should balance their presence equally in train val and test sets.
- Instead of fixed 0.5 class prediction threshold, each class could have a custom threshold that maximizes F1 score on test set.
- For evaluation, F1 score could be computed with a tolerance window
- Optimize model training to reduce RAM usage (right now all dataset constantly stays in RAM, I guess it's bottleneck for training speed)
