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
TODO

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
TODO F1 Score

## Possible Improvements
- Dataset audio samples belong to multiple classes (Drum kit, playing style). I should balance their presence equally in train val and test sets.
