import csv
import base64
import numpy as np
import matplotlib.pyplot as plt
import umap 
import mplcursors
import pyperclip
from pathlib import Path
from matplotlib import font_manager, rcParams
from matplotlib.widgets import CheckButtons
from sklearn.metrics.pairwise import cosine_similarity

font_path = r"C:\Windows\Fonts\YuGothR.ttc"
font_manager.fontManager.addfont(font_path)
rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()

# Load embeddings from CSV
embeddings_csv = "data/embeddings.csv"
window_ids = []
vectors = []
track_lookup = {}

with open("data/index.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        track_id = row["track_id"]
        path = Path(row.get("source_path") or row.get("original_path"))
        track_lookup[track_id] = path.stem  # filename without extension

labels = []
with open(embeddings_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        win_id = row["window_id"]
        window_ids.append(win_id)
        track_id = win_id.split("_")[0]
        window_type = win_id.split("_")[-1]
        title = track_lookup.get(track_id, track_id)
        labels.append(f"{title}\n[{window_type}]")
        emb_bytes = base64.b64decode(row["embedding"])
        vector = np.frombuffer(emb_bytes, dtype=np.float32)
        vectors.append(vector)

vectors = np.stack(vectors)

# Reduce to 2D
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding_2d = reducer.fit_transform(vectors)

# Separate data points by type
start_mask = np.array([wid.endswith("start") for wid in window_ids])
end_mask = np.array([wid.endswith("end") for wid in window_ids])
global_mask = ~(start_mask | end_mask)

# Plot
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(left=0.25, bottom=0.15)

# Create separate scatter plots for each type
sc_start = ax.scatter(
    embedding_2d[start_mask, 0],
    embedding_2d[start_mask, 1],
    c='C0',
    s=20,
    alpha=0.7,
    label='Start'
)

sc_end = ax.scatter(
    embedding_2d[end_mask, 0],
    embedding_2d[end_mask, 1],
    c='C1',
    s=20,
    alpha=0.7,
    label='End'
)

sc_global = ax.scatter(
    embedding_2d[global_mask, 0],
    embedding_2d[global_mask, 1],
    c='C2',
    s=20,
    alpha=0.7,
    label='Global'
)

# Store scatter plots and their indices
scatter_plots = {
    'Start': (sc_start, np.where(start_mask)[0]),
    'End': (sc_end, np.where(end_mask)[0]),
    'Global': (sc_global, np.where(global_mask)[0])
}

# State for cosine similarity mode
cosine_mode = {'enabled': False, 'selected_points': [], 'markers': [], 'text_annotation': None}

# Setup mplcursors for all scatter plots with hover=2 (hover mode with auto-removal)
cursors = []
for name, (sc, indices) in scatter_plots.items():
    cursor = mplcursors.cursor(sc, hover=2)  # hover=2 enables auto-removal on mouse leave
    
    @cursor.connect("add")
    def on_add(sel, indices=indices):
        original_index = indices[sel.index]
        sel.annotation.set_text(labels[original_index])
        sel.annotation.get_bbox_patch().set(alpha=0.9)
    
    # Add click handler to copy track name to clipboard
    def on_click(event, indices=indices):
        if event.artist in [sc]:
            # Get the index of the clicked point
            ind = event.index
            original_index = indices[ind]
            # Extract just the track name (first line before \n)
            track_name = labels[original_index].split('\n')[0]
            pyperclip.copy(track_name)
            print(f"Copied to clipboard: {track_name}")
    
    fig.canvas.mpl_connect('button_press_event', 
                          lambda event, idx=indices, s=sc: on_click_wrapper(event, idx, s))
    
    cursors.append(cursor)

# Wrapper to handle click events properly
def on_click_wrapper(event, indices, scatter):
    # Check if click is on a point in this scatter plot
    contains, details = scatter.contains(event)
    if contains:
        clicked_indices = details['ind']
        if len(clicked_indices) > 0:
            original_index = indices[clicked_indices[0]]
            
            # If cosine mode is enabled, handle point selection
            if cosine_mode['enabled']:
                handle_cosine_selection(original_index)
            else:
                # Normal mode: copy to clipboard
                track_name = labels[original_index].split('\n')[0]
                pyperclip.copy(track_name)
                print(f"Copied to clipboard: {track_name}")

def handle_cosine_selection(index):
    """Handle point selection in cosine similarity mode"""
    # Clear previous markers if we're starting fresh
    if len(cosine_mode['selected_points']) >= 2:
        for marker in cosine_mode['markers']:
            marker.remove()
        cosine_mode['markers'] = []
        cosine_mode['selected_points'] = []
        
        # Remove previous text annotation
        if cosine_mode['text_annotation'] is not None:
            cosine_mode['text_annotation'].remove()
            cosine_mode['text_annotation'] = None
    
    # Add the selected point
    cosine_mode['selected_points'].append(index)
    
    # Add a marker for the selected point
    x, y = embedding_2d[index]
    marker = ax.scatter([x], [y], c='red', s=100, marker='x', linewidths=3, zorder=10)
    cosine_mode['markers'].append(marker)
    
    print(f"Selected point {len(cosine_mode['selected_points'])}: {labels[index].split(chr(10))[0]}")
    
    # If we have two points, calculate cosine similarity
    if len(cosine_mode['selected_points']) == 2:
        idx1, idx2 = cosine_mode['selected_points']
        vec1 = vectors[idx1].reshape(1, -1)
        vec2 = vectors[idx2].reshape(1, -1)
        similarity = cosine_similarity(vec1, vec2)[0][0]
        
        track1 = labels[idx1].split('\n')[0]
        track2 = labels[idx2].split('\n')[0]
        
        result_text = f"Cosine Similarity: {similarity:.4f}\n{track1} ↔ {track2}"
        print("="*50)
        print(result_text)
        print("="*50)
        
        # Add text annotation on the plot
        cosine_mode['text_annotation'] = ax.text(0.02, 0.98, result_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.8), fontsize=10)
    
    plt.draw()

# Create checkboxes for visibility
rax = plt.axes([0.05, 0.6, 0.15, 0.15])
check = CheckButtons(rax, ['Start', 'End', 'Global'], [True, True, True])

# Create cosine similarity checkbox - positioned lower
rax_cosine = plt.axes([0.05, 0.05, 0.15, 0.08])
check_cosine = CheckButtons(rax_cosine, ['Cosine Mode'], [False])

def toggle_visibility(label):
    sc, _ = scatter_plots[label]
    sc.set_visible(not sc.get_visible())
    plt.draw()

def toggle_cosine_mode(label):
    cosine_mode['enabled'] = not cosine_mode['enabled']
    
    # Clear selections when toggling mode
    for marker in cosine_mode['markers']:
        marker.remove()
    cosine_mode['markers'] = []
    cosine_mode['selected_points'] = []
    
    # Clear any text annotations
    for txt in ax.texts[:]:
        if 'Cosine Similarity' in txt.get_text():
            txt.remove()
    
    mode_status = "ENABLED" if cosine_mode['enabled'] else "DISABLED"
    print(f"\nCosine Similarity Mode: {mode_status}")
    if cosine_mode['enabled']:
        print("Click on two points to calculate cosine similarity")
    else:
        print("Click mode: Copy track name to clipboard")
    
    plt.draw()

check.on_clicked(toggle_visibility)
check_cosine.on_clicked(toggle_cosine_mode)

ax.set_title("2D Visualisation of Music Embeddings")
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")
ax.legend()

plt.tight_layout()
plt.show()