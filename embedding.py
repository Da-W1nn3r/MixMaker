import csv
from pathlib import Path
import librosa
import openl3
import numpy as np
import base64

EMBEDDING_STRATEGY = "openl3_mel128_music"
EMBEDDING_VERSION = "v1"

def generate_embeddings(windows_csv, index_csv, embeddings_csv, sr=48000, embedding_size=512):
    windows_csv = Path(windows_csv)
    index_csv = Path(index_csv)
    embeddings_csv = Path(embeddings_csv)
    embeddings_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing embeddings to avoid duplicates
    existing_ids = set()
    existing_rows = []
    if embeddings_csv.exists():
        with embeddings_csv.open("r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            existing_ids = {row["window_id"] for row in reader}
            existing_rows = reader
    
    # Load track index
    with index_csv.open("r", encoding="utf-8") as f:
        index_rows = list(csv.DictReader(f))
    track_map = {row["track_id"]: row for row in index_rows}
    
    # Load windows
    with windows_csv.open("r", encoding="utf-8") as f:
        windows = list(csv.DictReader(f))
    
    new_rows = []
    total_windows = len(windows)
    progress = 0
    embeddings_processed = 0  # Count only new embeddings
    
    for win in windows:
        progress += 1
        win_id = win["window_id"]
        
        if win_id in existing_ids:
            continue  # Skip already processed windows
        
        print(f"Processing window {progress}/{total_windows}: {win_id} (embedded: {embeddings_processed})", end="\r")
        
        track = track_map[win["track_id"]]
        pcm_path = Path(track["pcm_path"])
        start = float(win["start_sec"])
        end = float(win["end_sec"])
        duration = end - start
        
        if duration <= 0:
            continue
        
        # Load audio segment
        y, _ = librosa.load(pcm_path, sr=sr, mono=True, offset=start, duration=duration)
        
        # Generate embedding
        emb, _ = openl3.get_audio_embedding(
            y,
            sr,
            input_repr="mel128",
            content_type="music",
            embedding_size=embedding_size
        )
        
        # Take mean across frames for window-level embedding
        window_emb = emb.mean(axis=0)
        
        # Convert to base64 string for CSV storage
        emb_bytes = window_emb.astype(np.float32).tobytes()
        emb_b64 = base64.b64encode(emb_bytes).decode("utf-8")
        
        new_rows.append({
            "window_id": win_id,
            "embedding": emb_b64,
            "embedding_strategy": EMBEDDING_STRATEGY,
            "embedding_version": EMBEDDING_VERSION,
        })
        existing_ids.add(win_id)
        embeddings_processed += 1
        
        # Save every 9 newly embedded windows
        if embeddings_processed % 9 == 0:
            all_rows = existing_rows + new_rows
            fieldnames = ["window_id", "embedding", "embedding_strategy", "embedding_version"]
            with embeddings_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)
            print(f"\nCheckpoint: Saved {embeddings_processed} embeddings")
    
    # Final save for any remaining embeddings
    if new_rows:
        all_rows = existing_rows + new_rows
        fieldnames = ["window_id", "embedding", "embedding_strategy", "embedding_version"]
        with embeddings_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
    
    print(f"\nEmbeddings generated for {len(new_rows)} new windows.")