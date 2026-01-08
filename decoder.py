import ffmpeg
import soundfile as sf
import numpy as np
import hashlib
import csv
from pathlib import Path

EMBEDDING_VERSION = "clap_v1_48k_mono"
CSV_FIELDS = [
    "track_id",
    "audio_hash",
    "original_path",
    "pcm_path",
    "duration_sec",
    "sample_rate",
    "channels",
    "embedding_version",
]

SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma",
    ".wav", ".flac", ".alac", ".aiff", ".aif", ".aifc",
    ".mod", ".xm", ".it", ".s3m",
}

SAMPLE_RATE = 48000
CHANNELS = 1

def hash_audio(audio: bytes) -> str:
    return hashlib.sha256(audio).hexdigest()

def compute_duration(num_samples: int, sample_rate: int) -> float:
    return num_samples / sample_rate

def load_existing_hashes(csv_path):
    hashes = set()
    if not csv_path.exists():
        return hashes

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hashes.add(row["audio_hash"])
    return hashes


def append_to_csv(csv_path, row, fieldnames):
    file_exists = csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def decode_to_wav(input_path, output_path):
    out, _ = (
        ffmpeg
        .input(str(input_path))
        .output(
            "pipe:",
            format="f32le",
            acodec="pcm_f32le",
            ac=CHANNELS,
            ar=SAMPLE_RATE
        )
        .run(capture_stdout=True, capture_stderr=True)
    )

    audio = np.frombuffer(out, dtype=np.float32)
    sf.write(output_path, audio, SAMPLE_RATE, subtype="FLOAT")
    return audio


def decode_folder(input_folder, output_folder, input_root=None):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    csv_path = Path("data/index.csv")
    existing_hashes = load_existing_hashes(csv_path)

    if input_root is None:
        input_root = input_folder

    for file in input_folder.iterdir():
        if file.is_dir():
            decode_folder(file, output_folder, input_root)
            continue

        if not file.is_file():
            continue

        if file.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue

        relative = file.relative_to(input_root)
        safe_name = "_".join(relative.with_suffix("").parts)
        output_file = output_folder / f"{safe_name}.wav"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            audio = decode_to_wav(file, output_file)
            audio_bytes = audio.tobytes()
            audio_hash = hash_audio(audio_bytes)

            if audio_hash in existing_hashes:
                print(f"Duplicate detected, skipping: {file}")
                output_file.unlink(missing_ok=True)
                continue

            duration = compute_duration(len(audio), SAMPLE_RATE)
            track_id = audio_hash[:16]

            row = {
                "track_id": track_id,
                "audio_hash": audio_hash,
                "original_path": str(file),
                "pcm_path": str(output_file),
                "duration_sec": round(duration, 3),
                "sample_rate": SAMPLE_RATE,
                "channels": CHANNELS,
                "embedding_version": EMBEDDING_VERSION,
            }

            append_to_csv(csv_path, row, CSV_FIELDS)
            existing_hashes.add(audio_hash)

            print(f"Decoded and indexed: {file}")

        except Exception as e:
            print(f"Error processing {file}: {e}")