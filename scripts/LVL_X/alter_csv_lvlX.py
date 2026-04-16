import pandas as pd
import os

# === SETTINGS ===
input_folder = "/Users/olivier/Documents/STREAMS/inD-dataset-v1.1_tracks_only"       # Folder with your original CSV files
output_folder = "/Users/olivier/Documents/STREAMS/inD_tracks_only_filtered"     # Folder where filtered files will be saved
columns_to_keep = ['trackId', 'frame', 'xCenter', 'yCenter']

# === MAKE OUTPUT FOLDER IF NEEDED ===
os.makedirs(output_folder, exist_ok=True)

# === PROCESS EACH CSV FILE ===
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            df = pd.read_csv(input_path)
            df_filtered = df[columns_to_keep]
            df_filtered.to_csv(output_path, index=False, header=False)
            print(f"✅ Saved cleaned file: {output_path}")
        except Exception as e:
            print(f"⚠️ Skipped {filename}: {e}")

print("\nAll done!")
