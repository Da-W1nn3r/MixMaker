import numpy as np

def softmin_walk(
    start_track,
    tracks,
    transition_cost,
    playlist_len,
    temperature=0.15,
    top_k=30
):
    playlist = [start_track]
    used = {start_track}
    
    print(f"  1. {start_track.original_path}")
    
    current = start_track

    for i in range(playlist_len - 1):
        # compute costs
        scored = []
        for t in tracks:
            if t in used:
                continue
            c = transition_cost(current, t)
            scored.append((t, c))

        # keep best K
        scored.sort(key=lambda x: x[1])
        scored = scored[:top_k]

        # softmin sampling
        costs = np.array([c for _, c in scored])
        probs = np.exp(-costs / temperature)
        probs /= probs.sum()

        next_track = np.random.choice(
            [t for t, _ in scored],
            p=probs
        )
        
        print(f"  {i+2}. {next_track.original_path}")

        playlist.append(next_track)
        used.add(next_track)
        current = next_track

    return playlist