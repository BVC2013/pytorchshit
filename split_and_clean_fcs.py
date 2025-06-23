import os
from fcsparser import parse
import pandas as pd

RAW_DIR = "data/raw_fcs"
SPECTRAL_DIR = "data/spectral_unmixed"
OTHER_DIR = "data/not_unmixed"

os.makedirs(SPECTRAL_DIR, exist_ok=True)
os.makedirs(OTHER_DIR, exist_ok=True)

def is_spectral_unmixed(meta):
    # Heuristic: look for a keyword in meta that indicates spectral unmixing
    # (e.g., 'SPILL' or 'SPECTRAL' or similar)
    for key in meta:
        if 'spectral' in key.lower() or 'unmix' in key.lower():
            return True
        if 'spill' in key.lower():
            return True
    return False

def remove_unwanted_headers(df):
    # Remove columns that are not fluorescence (same as detect_fluorescence_channels)
    keywords_to_exclude = ["FSC", "SSC", "Time", "Width", "Height", "Area", "Pulse"]
    keep_cols = [col for col in df.columns if all(k not in col for k in keywords_to_exclude)]
    return df[keep_cols]

for fname in os.listdir(RAW_DIR):
    if not fname.endswith(".fcs"): continue
    fpath = os.path.join(RAW_DIR, fname)
    meta, df = parse(fpath, reformat_meta=True)
    df_clean = remove_unwanted_headers(df)
    if is_spectral_unmixed(meta):
        out_path = os.path.join(SPECTRAL_DIR, fname.replace('.fcs', '.csv'))
    else:
        out_path = os.path.join(OTHER_DIR, fname.replace('.fcs', '.csv'))
    df_clean.to_csv(out_path, index=False)
print("Done splitting and cleaning FCS files.")
