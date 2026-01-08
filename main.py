from pathlib import Path
import decoder
import window
import metadata
import embedding

def main():
    print("Enter path to music folder:")
    input_path_str = input().strip()
    if not input_path_str:
        print("Skipping decoding new data")
    else:
        input_path = Path(input_path_str).expanduser().resolve()
        if not input_path.exists():
            print("Input path does not exist, skipping decoding new data")
            return
        else:
            output_path = Path("data/raw")
            output_path.mkdir(parents=True, exist_ok=True)
            decoder.decode_folder(input_path, output_path)
            
    print("Generating windows...")
    window.generate_windows(
        index_csv="data/index.csv",
        windows_csv="data/windows.csv"
    )
    print("Generating metadata...")
    metadata.extract_metadata(
        index_csv="data/index.csv", 
        windows_csv="data/windows.csv",
        metadata_csv="data/metadata.csv"
    )
    print("Generating embeddings...")
    embedding.generate_embeddings(
        windows_csv="data/windows.csv",
        index_csv="data/index.csv",
        embeddings_csv="data/embeddings.csv"
    )
    
if __name__ == "__main__":
    main()

# TODO: Implement visualisation of embeddings
# TODO: Implement playlist creation based on embeddings