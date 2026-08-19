# MixMaker

MixMaker builds DJ-style playlists from a folder of audio files. It decodes tracks to a
common format, extracts audio embeddings and metadata (tempo, key, energy), then walks
between tracks by picking low-cost transitions so that mixes flow smoothly from one song
to the next.

## How it works

The pipeline is split into a few stages, run in order from `main.py`:

1. **Decode** (`decoder.py`) — converts every audio file in an input folder (mp3, m4a,
   flac, wav, and more) to mono 48kHz WAV via `ffmpeg`, hashes the audio to skip
   duplicates, and records each track in `data/index.csv`.
2. **Window** (`window.py`) — trims silence from each track and defines three analysis
   windows per track: `start` (first ~15s), `end` (last ~15s), and `global` (a
   representative ~20-30s section from the middle). Written to `data/windows.csv`.
3. **Metadata** (`metadata.py`) — for each window, estimates BPM, musical key (via
   chroma), and an energy score using `librosa`. Written to `data/metadata.csv`.
4. **Embeddings** (`embedding.py`) — generates an [OpenL3](https://github.com/marl/openl3)
   audio embedding for each window, stored as base64-encoded float32 vectors in
   `data/embeddings.csv`.

Once the data is generated, `playlist.py` loads everything into `Track` objects and
builds a playlist:

- **Transition cost** (`cost.py`) — scores how well track B follows track A using a
  weighted combination of embedding similarity (start/global), tempo difference, energy
  transition (rewarding rising energy, penalizing drops), and key compatibility (circle
  of fifths).
- **Walk** (`walk.py`) — a softmin random walk: from the current track, the top-K lowest
  cost candidates are sampled with probability weighted by `exp(-cost / temperature)`,
  so lower temperatures produce more predictable/similar transitions and higher
  temperatures add variety.

The resulting playlist is exported as an `.m3u` or `.m3u8` file in `out/`.

`view.py` is a standalone tool for visualizing the embedding space: it reduces all
window embeddings to 2D with UMAP and renders an interactive scatter plot (hover to see
track names, click to copy a track name to the clipboard, or enable "Cosine Mode" to
compare the similarity of two selected points).

## Requirements

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/) installed and available on your `PATH`

Python dependencies (no `requirements.txt` is currently checked in — install these
manually):

```
ffmpeg-python
soundfile
numpy
pandas
librosa
openl3
umap-learn
matplotlib
mplcursors
pyperclip
scikit-learn
```

## Usage

Build the dataset and generate a playlist:

```
python main.py
```

You'll be prompted for a folder of music to decode (leave blank to skip decoding and
reuse existing data). This populates `data/index.csv`, `data/windows.csv`,
`data/metadata.csv`, and `data/embeddings.csv`.

Generate a playlist from the processed data:

```
python playlist.py
```

You'll be asked for a starting track (path, filename, or leave blank for random),
playlist length, temperature, top-K candidates, and an output filename. The playlist is
saved under `out/`.

Explore the embedding space visually:

```
python view.py
```

## Data layout

All generated data lives under `data/` and `out/`, both of which are gitignored:

- `data/index.csv` — one row per decoded track (id, paths, duration, hash)
- `data/windows.csv` — start/end/global time windows per track
- `data/metadata.csv` — BPM, key, key confidence, energy per window
- `data/embeddings.csv` — base64-encoded OpenL3 embedding per window
- `out/` — generated playlist files
