import os
import sys
import base64
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from cost import transition_cost
from walk import softmin_walk


@dataclass(frozen=True)
class Track:
    track_id: str
    original_path: str
    duration_sec: float
    
    # Start window
    start_emb: np.ndarray
    bpm: float
    key: str
    key_conf: float
    energy_start: float
    
    # Global window
    global_emb: np.ndarray
    energy_end: float
    
    def __hash__(self):
        return hash(self.track_id)
    
    def __eq__(self, other):
        if not isinstance(other, Track):
            return False
        return self.track_id == other.track_id


def load_data():
    """Load and merge all CSV data into Track objects."""
    
    # Load CSVs
    index_df = pd.read_csv("./data/index.csv")
    windows_df = pd.read_csv("./data/windows.csv")
    embeddings_df = pd.read_csv("./data/embeddings.csv")
    metadata_df = pd.read_csv("./data/metadata.csv")
    
    # Separate start and global windows
    start_windows = windows_df[windows_df["window_type"] == "start"]
    global_windows = windows_df[windows_df["window_type"] == "global"]
    
    # Merge start window data
    start_data = start_windows.merge(embeddings_df, on="window_id", how="inner")
    start_data = start_data.merge(metadata_df, on="window_id", how="inner")
    
    # Merge global window data
    global_data = global_windows.merge(embeddings_df, on="window_id", how="inner")
    global_data = global_data.merge(metadata_df, on="window_id", how="inner")
    
    # Create Track objects
    tracks = []
    
    for _, row in index_df.iterrows():
        track_id = row["track_id"]
        
        # Get start window data
        start_row = start_data[start_data["track_id"] == track_id]
        if start_row.empty:
            continue
        start_row = start_row.iloc[0]
        
        # Get global window data
        global_row = global_data[global_data["track_id"] == track_id]
        if global_row.empty:
            continue
        global_row = global_row.iloc[0]
        
        # Decode embeddings
        start_emb = np.frombuffer(base64.b64decode(start_row["embedding"]), dtype=np.float32)
        global_emb = np.frombuffer(base64.b64decode(global_row["embedding"]), dtype=np.float32)
        
        # Normalize embeddings
        start_emb = start_emb / (np.linalg.norm(start_emb) + 1e-8)
        global_emb = global_emb / (np.linalg.norm(global_emb) + 1e-8)
        
        track = Track(
            track_id=track_id,
            original_path=row["original_path"],
            duration_sec=row["duration_sec"],
            start_emb=start_emb,
            bpm=start_row["bpm"],
            key=start_row["key"],
            key_conf=start_row["key_confidence"],
            energy_start=start_row["energy"],
            global_emb=global_emb,
            energy_end=global_row["energy"]
        )
        
        tracks.append(track)
    
    return tracks


def find_track(tracks, user_input):
    """Find a track based on user input (path or filename)."""
    
    # Remove quotes if present
    user_input = user_input.strip().strip('"').strip("'")
    
    # Try exact path match first
    for track in tracks:
        if track.original_path == user_input:
            return track
    
    # Try filename match (with or without extension)
    user_path = Path(user_input)
    user_stem = user_path.stem.lower()
    user_name = user_path.name.lower()
    
    for track in tracks:
        track_path = Path(track.original_path)
        if track_path.name.lower() == user_name or track_path.stem.lower() == user_stem:
            return track
    
    return None


def create_m3u8(playlist, output_path):
    """Create an M3U8 playlist file."""
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("#EXTM3U\n")
        
        for track in playlist:
            duration = int(track.duration_sec)
            title = Path(track.original_path).stem
            
            f.write(f"#EXTINF:{duration},{title}\n")
            f.write(f"{track.original_path}\n")
    
    print(f"\n✓ Playlist saved to: {output_path}")


def create_m3u(playlist, output_path):
    """Create an M3U playlist file (simple format without metadata)."""
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for track in playlist:
            f.write(f"{track.original_path}\n")
    
    print(f"\n✓ Playlist saved to: {output_path}")


def get_user_input(prompt, default=None, input_type=str):
    """Get user input with optional default value."""
    
    if default is not None:
        prompt = f"{prompt} [default: {default}]: "
    else:
        prompt = f"{prompt}: "
    
    user_input = input(prompt).strip()
    
    if not user_input and default is not None:
        return default
    
    if not user_input and default is None:
        return None
    
    try:
        return input_type(user_input)
    except ValueError:
        print(f"Invalid input. Using default: {default}")
        return default


def main():
    print("=" * 60)
    print("  PLAYLIST GENERATOR")
    print("=" * 60)
    
    # Load all track data
    print("\nLoading track data...")
    tracks = load_data()
    print(f"✓ Loaded {len(tracks)} tracks")
    
    # Get starting track
    print("\n" + "-" * 60)
    print("STARTING TRACK")
    print("-" * 60)
    print("Enter a song in any of these formats:")
    print('  - Full path: "D:\\Music\\Artist\\Song.mp3"')
    print("  - Full path: D:\\Music\\Artist\\Song.mp3")
    print("  - Filename: Song.mp3")
    print("  - Name only: Song")
    print("  - Leave empty for random selection")
    
    start_input = input("\nStarting song: ").strip()
    
    if not start_input:
        start_track = np.random.choice(tracks)
        print(f"✓ Random selection: {Path(start_track.original_path).name}")
    else:
        start_track = find_track(tracks, start_input)
        if start_track is None:
            print(f"\n✗ Track not found: {start_input}")
            print("Using random track instead...")
            start_track = np.random.choice(tracks)
            print(f"✓ Selected: {Path(start_track.original_path).name}")
        else:
            print(f"✓ Found: {Path(start_track.original_path).name}")
    
    # Get playlist parameters
    print("\n" + "-" * 60)
    print("PLAYLIST PARAMETERS")
    print("-" * 60)
    
    playlist_len = get_user_input(
        "Playlist length",
        default=25,
        input_type=int
    )
    
    print("\nTemperature (similarity):")
    print("  - Lower (0.1-0.2): Very similar songs")
    print("  - Medium (0.3-0.5): Balanced variety")
    print("  - Higher (0.6-1.0): More exploration")
    
    temperature = get_user_input(
        "Temperature",
        default=0.15,
        input_type=float
    )
    
    top_k = get_user_input(
        "Top K candidates to consider",
        default=30,
        input_type=int
    )
    
    output_file = get_user_input(
        "Output file",
        default="playlist.m3u8"
    )
    
    # Ensure output is in ./out directory
    if not output_file.startswith("out"):
        output_file = os.path.join("out", output_file)
    
    # Determine file format
    output_ext = Path(output_file).suffix.lower()
    if output_ext not in ['.m3u8', '.m3u']:
        print(f"Warning: Unknown extension '{output_ext}', defaulting to .m3u8 format")
        output_ext = '.m3u8'
    
    # Generate playlist
    print("\n" + "-" * 60)
    print("GENERATING PLAYLIST")
    print("-" * 60)
    print(f"Starting from: {Path(start_track.original_path).name}")
    print(f"Target length: {playlist_len} songs")
    print(f"Temperature: {temperature}")
    print(f"Top K: {top_k}\n")
    
    playlist = softmin_walk(
        start_track=start_track,
        tracks=tracks,
        transition_cost=transition_cost,
        playlist_len=playlist_len,
        temperature=temperature,
        top_k=top_k
    )
    
    print(f"\n✓ Generated playlist with {len(playlist)} tracks")
    
    # Create playlist file
    if output_ext == '.m3u':
        create_m3u(playlist, output_file)
    else:
        create_m3u8(playlist, output_file)
    
    print("\n" + "=" * 60)
    print("  DONE!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        