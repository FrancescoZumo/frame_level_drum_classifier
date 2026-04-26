from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import onnxruntime as ort
import numpy as np
import librosa
import tempfile
import os

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# load model
SESSION = ort.InferenceSession("drum_cnn.onnx", providers=["CPUExecutionProvider"])
CONTEXT  = 3
SR       = 22050
HOP      = 256
N_FFT    = 1024
N_MELS   = 80
CLASSES  = ["kick", "snare", "hihat"]

# define API routes FIRST
@app.post("/transcribe")
async def transcribe(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        y, _ = librosa.load(tmp_path, sr=SR, mono=True)
        mel     = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT,
                                                  hop_length=HOP, n_mels=N_MELS)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        delta   = librosa.feature.delta(log_mel)
        delta2  = librosa.feature.delta(log_mel, order=2)
        features = np.stack([log_mel, delta, delta2], axis=0).astype(np.float32)

        T = features.shape[-1]
        pad = np.zeros((3, N_MELS, CONTEXT), dtype=np.float32)
        fp  = np.concatenate([pad, features, pad], axis=2)
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(fp, 2*CONTEXT+1, axis=2)
        windows = windows.transpose(2, 0, 3, 1).astype(np.float32)

        all_probs = []
        for i in range(0, T, 4096):
            batch  = windows[i:i+4096]
            logits = SESSION.run(None, {"input": batch})[0]
            all_probs.append(1 / (1 + np.exp(-logits)))
        probs = np.concatenate(all_probs, axis=0)
        preds = (probs > 0.5).astype(np.float32)

        onsets = {cls: [] for cls in CLASSES}
        for cls_idx, cls in enumerate(CLASSES):
            col = preds[:, cls_idx]
            for i in range(1, len(col)):
                if col[i] == 1 and col[i-1] == 0:
                    onsets[cls].append(round(i * HOP / SR, 4))

        duration = len(y) / SR
        return {"onsets": onsets, "duration": duration}

    finally:
        os.unlink(tmp_path)

# mount static files LAST — after all routes are defined
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# uvicorn server:app --reload