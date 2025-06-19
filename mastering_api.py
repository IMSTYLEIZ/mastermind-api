from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os
import librosa
import numpy as np
import pyloudnorm as pyln

app = FastAPI()

def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=None, mono=False)
    if y.ndim == 1:
        mono = True
        y_mono = y
    else:
        mono = False
        y_mono = librosa.to_mono(y)

    duration = librosa.get_duration(y=y, sr=sr)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y_mono)
    peak_db = 20 * np.log10(np.max(np.abs(y_mono)))
    rms = np.sqrt(np.mean(y_mono**2))
    rms_db = 20 * np.log10(rms)
    spectral_centroid = librosa.feature.spectral_centroid(y=y_mono, sr=sr)
    mean_centroid = np.mean(spectral_centroid)
    onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr)
    transients = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    stereo_width = np.std(y[0] - y[1]) if not mono else 0

    return {
        "duration_sec": round(duration, 2),
        "loudness_LUFS": round(loudness, 2),
        "peak_dBFS": round(peak_db, 2),
        "rms_dBFS": round(rms_db, 2),
        "mean_spectral_centroid": round(mean_centroid, 2),
        "num_transients": len(transients),
        "stereo_width_index": round(stereo_width, 4),
        "is_mono": mono
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = analyze_audio(temp_path)
        os.remove(temp_path)
        return JSONResponse(content=result)
    except Exception as e:
        os.remove(temp_path)
        return JSONResponse(status_code=500, content={"error": str(e)})