import csv
from pathlib import Path
import librosa

WINDOW_SEC = 15.0
SILENCE_DB = 25

GLOBAL_MAX_SEC = 30.0
GLOBAL_MIN_SEC = 20.0

WINDOW_STRATEGY = "silence_trimmed_fixed"
WINDOW_VERSION = "v2"

def generate_windows(index_csv, windows_csv):
    index_csv = Path(index_csv)
    windows_csv = Path(windows_csv)
    windows_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load existing window IDs to avoid duplicates
    existing_ids = set()
    if windows_csv.exists():
        with windows_csv.open("r", newline="", encoding="utf-8") as f:
            existing_ids = {row["window_id"] for row in csv.DictReader(f)}

    # Read track index
    if not index_csv.exists():
        print(f"Index file not found: {index_csv}")
        return

    with index_csv.open("r", newline="", encoding="utf-8") as f:
        tracks = list(csv.DictReader(f))

    if not tracks:
        print(f"No tracks found in index: {index_csv}")
        return

    fieldnames = [
        "window_id",
        "track_id",
        "window_type",
        "start_sec",
        "end_sec",
        "strategy",
        "strategy_version",
    ]

    rows = []
    print(f"Generating windows for {len(tracks)} tracks...")

    for i, track in enumerate(tracks, 1):
        print(f"  Processing track {i}/{len(tracks)}: {track['track_id']}", end="\r")
        track_id = track["track_id"]
        pcm_path = Path(track["pcm_path"])
        sr = int(track["sample_rate"])

        if not pcm_path.exists():
            print(f"\n  Skipping missing file: {pcm_path}")
            continue

        # Load audio
        y, sr = librosa.load(pcm_path, sr=sr, mono=True)

        # Silence trim
        y_trim, (start_sample, end_sample) = librosa.effects.trim(y, top_db=SILENCE_DB)
        trim_start = start_sample / sr
        trim_end = end_sample / sr
        trimmed_duration = trim_end - trim_start
        if trimmed_duration <= 0:
            continue

        # ---- START window ----
        start_win_id = f"{track_id}_start"
        if start_win_id not in existing_ids:
            start_win_end = min(trim_start + WINDOW_SEC, trim_end)
            rows.append({
                "window_id": start_win_id,
                "track_id": track_id,
                "window_type": "start",
                "start_sec": round(trim_start, 3),
                "end_sec": round(start_win_end, 3),
                "strategy": WINDOW_STRATEGY,
                "strategy_version": WINDOW_VERSION,
            })
            existing_ids.add(start_win_id)

        # ---- END window ----
        end_win_id = f"{track_id}_end"
        if end_win_id not in existing_ids:
            end_win_start = max(trim_end - WINDOW_SEC, trim_start)
            rows.append({
                "window_id": end_win_id,
                "track_id": track_id,
                "window_type": "end",
                "start_sec": round(end_win_start, 3),
                "end_sec": round(trim_end, 3),
                "strategy": WINDOW_STRATEGY,
                "strategy_version": WINDOW_VERSION,
            })
            existing_ids.add(end_win_id)

        # ---- GLOBAL window ----
        global_win_id = f"{track_id}_global"
        if trimmed_duration >= GLOBAL_MIN_SEC and global_win_id not in existing_ids:
            # Calculate middle section avoiding start/end windows
            middle_start = trim_start + WINDOW_SEC  # After start window
            middle_end = trim_end - WINDOW_SEC      # Before end window
            middle_duration = middle_end - middle_start
            
            if middle_duration >= GLOBAL_MIN_SEC:
                # There's enough space in the middle for non-overlapping window
                if middle_duration <= GLOBAL_MAX_SEC:
                    # Use entire middle section
                    global_start = middle_start
                    global_end = middle_end
                else:
                    # Center it in the middle section
                    centre = (middle_start + middle_end) / 2
                    half = GLOBAL_MAX_SEC / 2
                    global_start = centre - half
                    global_end = centre + half
            else:
                # Not enough space in middle, fallback to centered approach
                if trimmed_duration <= GLOBAL_MAX_SEC:
                    global_start = trim_start
                    global_end = trim_end
                else:
                    centre = (trim_start + trim_end) / 2
                    half = GLOBAL_MAX_SEC / 2
                    global_start = centre - half
                    global_end = centre + half

            rows.append({
                "window_id": global_win_id,
                "track_id": track_id,
                "window_type": "global",
                "start_sec": round(global_start, 3),
                "end_sec": round(global_end, 3),
                "strategy": WINDOW_STRATEGY,
                "strategy_version": WINDOW_VERSION,
            })
            existing_ids.add(global_win_id)

    # Write new rows (append mode)
    if rows:
        file_exists = windows_csv.exists()
        mode = "a" if file_exists else "w"
        with windows_csv.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

    print(f"\nDone. {len(rows)} new windows written to {windows_csv}")