import os
import pandas as pd
import numpy as np
import librosa
from time import time
from concurrent.futures import ProcessPoolExecutor, as_completed

#region params


# path to locally stored dataset
# TRAINING DATASET: STAR Dataset, https://zenodo.org/records/15690078 
TRAIN_DATASET_PATH = "D:\\Downloads\\STAR_Drums_full\\STAR_publication\\data\\training"
VALIDATION_DATASET_PATH = "D:\\Downloads\\STAR_Drums_full\\STAR_publication\\data\\validation"
TEST_DATASET_PATH = "D:\\Downloads\\STAR_Drums_full\\STAR_publication\\data\\test"

# Audio global parameters
AUDIO_FILE_DURATION = 60 #seconds
SR = 22050
HOP_LENGTH = 256   # hop_length is ~10 ms time resolution
N_MELS = 96        # standard, captures enough spectral detail
N_FFT = 1024        # ~46 ms time resolution, 21Hz freq resolution, TODO if too low increase to 2048 


PRECOMPUTED_FEATURES_FOLDER = "precomputed_features"
TRAIN_FEATURES = os.path.join(PRECOMPUTED_FEATURES_FOLDER, f"sr{SR}_hop{HOP_LENGTH}_nfft{N_FFT}_mels{N_MELS}")
VALIDATION_FEATURES = os.path.join(PRECOMPUTED_FEATURES_FOLDER, f"sr{SR}_hop{HOP_LENGTH}_nfft{N_FFT}_mels{N_MELS}", "validation")
TEST_FEATURES = os.path.join(PRECOMPUTED_FEATURES_FOLDER, f"sr{SR}_hop{HOP_LENGTH}_nfft{N_FFT}_mels{N_MELS}", "test")

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

TARGET_CLASSES = ['kick', 'snare', 'hihat']


#endregion

#region feature extraction

def get_frame_level_annotations(df, audio_duration=AUDIO_FILE_DURATION, sr=SR, hop_length=HOP_LENGTH,
                           active_duration=0.05):
    """
    df: dataframe with columns [timestamp, class, velocity] \n
    audio_duration: length of the audio file in seconds\n
    active_duration: how long a drum hit is considered "active" in seconds. \n
    For now we assume that the total ASDR of a hit is 50ms, though this could be improved and specified per each class
    
    Returns: np.array of shape (n_frames, 3)
    """
    # column names
    df.columns = ['time', 'class']

    # initialize empty array for annotations
    n_frames = int(np.ceil(audio_duration * sr / hop_length))
    y = np.zeros((n_frames, len(TARGET_CLASSES)), dtype=np.float32)

    # how many frames does the sould of a hit last? for now fixed for all classes
    active_frames = max(1, int(active_duration * sr / hop_length))


    for _, row in df.iterrows():
        mapped = CLASS_MAP.get(row['class'])
        if mapped is None:
            continue  # ignore toms, cymbals, etc.
        
        # detect onset frame of hit
        frame_idx = round(row['time'] * sr / hop_length) # convert time annotation from seconds to frame
        cls_idx = TARGET_CLASSES.index(mapped)

        # propagate onset annotation for fixed duration (active_frames)
        start = frame_idx
        end = min(frame_idx + active_frames, n_frames)
        y[start:end, cls_idx] = 1.0

    return y


def extract_audio_features(audio_path: str):

    #start = time()

    y, _ = librosa.load(audio_path, sr=SR, mono=True) # for fast feature extraction, process tracks in mono and 22050 SR, enough for drums transcription

    # for the first attempt I will use a simple CNN, no sequence model, 
    # so I use derivatives to give more context to current frame

    # 1st feature : log mel spectrgram
    # NOTE: I could Use CTQ which gives more frequency resolution to low frequencies than high frequencies,
    # but it's slightly slower and less straightforward to reproduce outside of python
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, 
                                      hop_length=HOP_LENGTH, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # 2nd feature: first derivative of mel spectrogram
    delta = librosa.feature.delta(log_mel)
    # 3rd feature: second derivative of mel spectrogram
    delta2 = librosa.feature.delta(log_mel, order=2)

    features = np.stack([log_mel, delta, delta2], axis=0)  # 3 channels

    duration = len(y) / SR 

    return features, duration



def extract_windows(features, labels, context=7): # used to provide chunks of frames to the CNN, like if audio was an image
    """
    features: (3, 80, T) 
    labels:   (T, 3)
    returns X: (T, 3, 15, 80), y: (T, 3)
    """
    features = features.transpose(2, 0, 1)  # (T, 3, 80)
    n_frames = len(labels)
    pad = np.zeros((context, 3, N_MELS), dtype=np.float32)
    features_padded = np.concatenate([pad, features, pad], axis=0)  # (T+2*context, 3, 80)

    X = np.stack([features_padded[i:i + 2*context + 1] for i in range(n_frames)], axis=0)  # (T, 15, 3, 80)
    
    # reshape to (T, 3, 15, 80) for pytorch Conv2d: (batch, channels, height, width)
    X = X.transpose(0, 2, 1, 3)
    return X, labels


def extract_features_and_labels(args):
    """
    Extract features and labels from single track
    """

    file, audio_files_path, annotation_files_path = args

    file_id = file.split("_")[1]


    audio_path = os.path.join(audio_files_path, file.replace(".txt", ".flac"))
    annot_path = os.path.join(annotation_files_path, file)

    if not os.path.exists(audio_path):
        # names could not be the same btw annotation and audio files, I can look for ID for match
        available_audios = os.listdir(audio_files_path)
        files_with_same_id = [f for f in available_audios if file_id in f]
        if len(files_with_same_id) > 1:
            print("more than one files (or none) with same Id: {}, skipping".format(files_with_same_id))
            return None
        elif len(files_with_same_id) == 0:
            print(f"Missing audio for {file}, skipping")
            return None
        # Found exactly one match --> this is the correct audio file
        audio_path = os.path.join(audio_files_path, files_with_same_id[0])
        # check if file exists
        if not os.path.exists(audio_path):
            print(f"Missing audio for {file}, skipping")
            return None

    try:
        features, duration = extract_audio_features(audio_path)
        annotations = pd.read_csv(annot_path, sep="\t", header=None,
                                    names=['time', 'class', 'velocity'])
        # remove unused velocity column
        annotations = annotations.drop(columns=['velocity'])
        labels = get_frame_level_annotations(annotations, audio_duration=duration)

        T = min(features.shape[-1], labels.shape[0])
        features = features[:, :, :T]
        labels = labels[:T]

        return (features, labels)

    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None
    
def load_tracks(train_annotations_path, train_audios_path, max_elements=None, n_workers=4):
    annotation_files = sorted(os.listdir(train_annotations_path))

    if max_elements is not None:
        annotation_files = annotation_files[:max_elements]

    # build args list for each worker
    args_list = [
        (file, train_audios_path, train_annotations_path)
        for file in annotation_files
    ]

    tracks = []
    completed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(extract_features_and_labels, args): args[0] 
                   for args in args_list}
        
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if result is not None:
                tracks.append(result)
            if completed % 100 == 0:
                print(f"Loaded {completed}/{len(annotation_files)} tracks")

    return tracks

def load_training_data(max_elements: int = None, n_workers: int = 4, load_if_available: bool = True):
    """
    Prepares training data as a list of (features, labels) tuples, one per track.
    This keeps tracks separate, making it compatible with both CNN and RNN pipelines.
    
    For CNN: concatenate and shuffle across all frames
    For RNN: iterate track by track, preserving temporal order

    - Uses multiprocessing for parallel feature extraction.

    - Loads from cache if available, otherwise extracts features and saves them to file.

    """

    if load_if_available and os.path.exists(TRAIN_FEATURES) and len(os.listdir(TRAIN_FEATURES)) > 0:
        print("Cache found, loading from disk...")
        tracks = load_paired_tracks_from_cache(TRAIN_FEATURES)
        if max_elements is not None:
            print(f"Limiting to first {max_elements} tracks from cache")
            tracks = tracks[:max_elements]
        return tracks

    # extract train features from mix folder
    train_audios_path = os.path.join(TRAIN_DATASET_PATH, "ismir04\\audio\\mix")
    train_annotations_path = os.path.join(TRAIN_DATASET_PATH, "ismir04\\annotation")
    
    mix_tracks = load_tracks(train_annotations_path, train_audios_path, max_elements=max_elements, n_workers=n_workers)

    # extract train features from drums_resynthesized folder
    train_audios_path_2 = os.path.join(TRAIN_DATASET_PATH, "ismir04\\audio\\re_synthesized_drum")
    train_annotations_path_2 = os.path.join(TRAIN_DATASET_PATH, "ismir04\\annotation")
    
    resyn_tracks = load_tracks(train_annotations_path_2, train_audios_path_2, max_elements=max_elements, n_workers=n_workers)

    # pair tracks by index to prevent leaage during split
    paired_tracks = list(zip(mix_tracks, resyn_tracks))

    print(f"Loaded {len(paired_tracks)} track pairs")
    save_paired_tracks(paired_tracks, TRAIN_FEATURES)
    return paired_tracks
    # save_tracks(validation_tracks, VALIDATION_FEATURES)
    # save_tracks(test_tracks, TEST_FEATURES)

    return all_tracks


def prepare_for_cnn(tracks, context=7):
    """Concatenates all windows across tracks — order doesn't matter for CNN."""
    X_all, y_all = [], []
    for features, labels in tracks:
        X, y = extract_windows(features, labels, context=context)
        X_all.append(X)
        y_all.append(y)
    
    X = np.concatenate(X_all, axis=0)  # (total_frames, 3, 2*context+1, 80)
    y = np.concatenate(y_all, axis=0)  # (total_frames, 3)
    
    # shuffle — safe for CNN since windows are independent
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def prepare_for_rnn(tracks, chunk_size=256):
    """
    Splits each track into fixed-length chunks — temporal order preserved.
    chunk_size: number of frames per chunk (e.g. 256 frames ~ 3 seconds)
    """
    X_all, y_all = [], []
    for features, labels in tracks:
        # features: (3, 80, T) → (T, 3, 80)
        features = features.transpose(2, 0, 1)
        T = len(labels)
        
        # split into chunks of chunk_size, drop last incomplete chunk
        for start in range(0, T - chunk_size + 1, chunk_size):
            X_all.append(features[start:start + chunk_size])   # (chunk_size, 3, 80)
            y_all.append(labels[start:start + chunk_size])     # (chunk_size, 3)
    
    return np.array(X_all), np.array(y_all)  # (n_chunks, chunk_size, 3, 80), (n_chunks, chunk_size, 3)

#endregion

#region LOAD FROM FILE

def save_tracks(tracks, cache_path=TRAIN_FEATURES):
    """Save tracks to disk as numpy arrays."""
    os.makedirs(cache_path, exist_ok=True)
    for i, (features, labels) in enumerate(tracks):
        np.save(os.path.join(cache_path, f"track_{i:04d}_features.npy"), features)
        np.save(os.path.join(cache_path, f"track_{i:04d}_labels.npy"), labels)
    print(f"Saved {len(tracks)} tracks to {cache_path}")


def load_tracks_from_cache(cache_path=TRAIN_FEATURES):
    """Load tracks from disk."""
    feature_files = sorted([f for f in os.listdir(cache_path) if f.endswith("_features.npy")])
    tracks = []
    for f in feature_files:
        features = np.load(os.path.join(cache_path, f))
        labels   = np.load(os.path.join(cache_path, f.replace("_features.npy", "_labels.npy")))
        tracks.append((features, labels))
    print(f"Loaded {len(tracks)} tracks from cache.")
    return tracks



def save_paired_tracks(paired_tracks, cache_path=TRAIN_FEATURES):
    """Save paired (mix, resynthesized) tracks to disk."""
    os.makedirs(cache_path, exist_ok=True)
    for i, (mix, resyn) in enumerate(paired_tracks):
        mix_features, mix_labels     = mix
        resyn_features, resyn_labels = resyn
        np.save(os.path.join(cache_path, f"track_{i:04d}_mix_features.npy"),   mix_features)
        np.save(os.path.join(cache_path, f"track_{i:04d}_labels.npy"),     mix_labels)
        np.save(os.path.join(cache_path, f"track_{i:04d}_resyn_features.npy"), resyn_features)
    print(f"Saved {len(paired_tracks)} track pairs to {cache_path}")


def load_paired_tracks_from_cache(cache_path=TRAIN_FEATURES):
    """Load paired tracks from disk."""
    n_tracks = len([f for f in os.listdir(cache_path) if f.endswith('_mix_features.npy')])
    paired_tracks = []
    for i in range(n_tracks):
        mix_features   = np.load(os.path.join(cache_path, f"track_{i:04d}_mix_features.npy"))
        labels: np.ndarray = np.load(os.path.join(cache_path, f"track_{i:04d}_labels.npy"))
        resyn_features = np.load(os.path.join(cache_path, f"track_{i:04d}_resyn_features.npy"))
        paired_tracks.append(((mix_features, labels), (resyn_features, labels.copy())))
    print(f"Loaded {len(paired_tracks)} track pairs from cache.")
    return paired_tracks


#endregion