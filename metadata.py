import csv
from pathlib import Path
import librosa
import numpy as np

METADATA_STRATEGY = "librosa_basic"
METADATA_VERSION = "v1"

def estimate_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    key_index = int(np.argmax(chroma_mean))
    confidence = chroma_mean[key_index] / (chroma_mean.sum() + 1e-6)
    return key_index, float(confidence)

def estimate_energy(y, sr):
    rms = librosa.feature.rms(y=y).mean()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    centroid_norm = centroid / (sr / 2)
    return float(rms * centroid_norm)

def extract_metadata(index_csv, windows_csv, metadata_csv):
    index_csv = Path(index_csv)
    windows_csv = Path(windows_csv)
    metadata_csv = Path(metadata_csv)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load existing metadata to avoid duplicates
    existing_ids = set()
    if metadata_csv.exists():
        with metadata_csv.open("r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            existing_ids = {row["window_id"] for row in reader}
    else:
        reader = []

    # Load track index
    if not index_csv.exists():
        print(f"Index file not found: {index_csv}")
        return
    with index_csv.open("r", encoding="utf-8") as f:
        index_rows = list(csv.DictReader(f))
    track_map = {row["track_id"]: row for row in index_rows}

    # Load windows
    if not windows_csv.exists():
        print(f"Windows file not found: {windows_csv}")
        return
    with windows_csv.open("r", encoding="utf-8") as f:
        windows = list(csv.DictReader(f))

    if not windows:
        print("No windows found to process")
        return

    rows = []
    print(f"Processing metadata for {len(windows)} windows...")

    for i, win in enumerate(windows, 1):
        print(f"  Window {i}/{len(windows)}: {win['window_id']}", end="\r")
        win_id = win["window_id"]
        if win_id in existing_ids:
            continue

        track = track_map.get(win["track_id"])
        if not track:
            print(f"\n  Skipping window {win_id}: track not found")
            continue

        pcm_path = Path(track["pcm_path"])
        sr = int(track["sample_rate"])
        start = float(win["start_sec"])
        end = float(win["end_sec"])
        duration = end - start
        if duration <= 0 or not pcm_path.exists():
            continue

        # Load audio segment
        y, sr = librosa.load(
            pcm_path,
            sr=sr,
            mono=True,
            offset=start,
            duration=duration
        )

        # Extract features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        key, key_conf = estimate_key(y, sr)
        energy = estimate_energy(y, sr)

        rows.append({
            "window_id": win_id,
            "bpm": round(float(tempo), 2),
            "key": key,
            "key_confidence": round(key_conf, 3),
            "energy": round(energy, 6),
            "metadata_strategy": METADATA_STRATEGY,
            "metadata_version": METADATA_VERSION,
        })
        existing_ids.add(win_id)

    # Write new rows (append mode)
    if rows:
        file_exists = metadata_csv.exists()
        mode = "a" if file_exists else "w"
        fieldnames = [
            "window_id",
            "bpm",
            "key",
            "key_confidence",
            "energy",
            "metadata_strategy",
            "metadata_version",
        ]
        with metadata_csv.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

    print(f"\nDone. {len(rows)} new metadata rows written to {metadata_csv}")
