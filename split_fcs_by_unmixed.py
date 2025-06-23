import os
from fcsparser import parse
import pandas as pd
from fcswrite import write_fcs

FILES_DIR = "files"
RAW_FCS_DIR = "data/raw_fcs"
UNMIXED_DIR = "files/unmixed_fcs"

os.makedirs(RAW_FCS_DIR, exist_ok=True)
os.makedirs(UNMIXED_DIR, exist_ok=True)

# Process all FCS files in the files directory
for fname in os.listdir(FILES_DIR):
    if not fname.lower().endswith('.fcs'):
        continue
    input_fcs = os.path.join(FILES_DIR, fname)
    base = os.path.splitext(fname)[0]
    unmixed_out = os.path.join(UNMIXED_DIR, f"{base}_unmixed.fcs")
    not_unmixed_out = os.path.join(RAW_FCS_DIR, f"{base}_not_unmixed.fcs")

    try:
        meta, df = parse(input_fcs, reformat_meta=True)
    except Exception as e:
        print(f"Skipping {fname}: {e}")
        continue

    unmixed_cols = [col for col in df.columns if 'spectral' in col.lower()]
    not_unmixed_cols = [col for col in df.columns if 'spectral' not in col.lower()]

    def remove_unwanted_headers(df):
        keywords_to_exclude = ["FSC", "SSC", "Time", "Width", "Height", "Area", "Pulse"]
        keep_cols = [col for col in df.columns if all(k not in col for k in keywords_to_exclude)]
        return df[keep_cols]

    unmixed_df = remove_unwanted_headers(df[unmixed_cols])
    not_unmixed_df = remove_unwanted_headers(df[not_unmixed_cols])

    if not unmixed_df.empty:
        write_fcs(unmixed_out, unmixed_df.columns.tolist(), unmixed_df.values)
    if not not_unmixed_df.empty:
        write_fcs(not_unmixed_out, not_unmixed_df.columns.tolist(), not_unmixed_df.values)

print("Done splitting all FCS files. Not-unmixed files are in data/raw_fcs, unmixed files are in files/unmixed_fcs.")
