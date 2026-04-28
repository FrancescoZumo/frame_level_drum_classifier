# Frame Level Drum Classifier
I trained a lightweight CNN to transcribe kicks, snares and hi-hats from any input audio file.

## Setup instructions
- Install requirements
```
conda create --name drums_classifier --file reqirements.txt
```

- run the web UI demo to try the model exported to onnx
```
cd webUI
uvicorn server:app --reload
```


## Method Overview
TODO

## Training Dataset
 - STAR Dataset, https://zenodo.org/records/15690078.
Among all analyzed options, This dataset provides drums annotations for both drums only recordings and complete song mix.
Also, the annotation schema is relatively simple to parse, with respect to other datasets (e.g. IDMT-SMT-Drums https://www.idmt.fraunhofer.de/en/publications/datasets/drums.html)

## Model architecture
I chose to implement a simple, lightweight CNN, starting from what was used in paper https://ismir2025program.ismir.net/poster_130.html 




## Design decisions and trade-offs

- Feature Extraction
I used mel-spectrograms as main feature.

Input CNN features 
TODO


I reduced the the number of convolutional blocks to 3, and the context window from 25 to 5

TODO

## Evaluation results
TODO F1 Score

## Possible Improvements
- Dataset audio samples belong to multiple classes (Drum kit, playing style). I should balance their presence equally in train val and test sets.
