import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from fcsparser import parse
import torch

def load_fcs(file_path):
    meta, df = parse(file_path, reformat_meta=True)
    return df, meta

def detect_fluorescence_channels(df, meta):
    keywords_to_exclude = ["FSC", "SSC", "Time", "Width", "Height", "Area", "Pulse"]
    fluorescence_cols = [
        col for col in df.columns
        if all(k not in col for k in keywords_to_exclude)
    ]
    return df[fluorescence_cols]

def arcsinh_transform(df, cofactor=150.0):
    return np.arcsinh(df / cofactor)

def normalize(df):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

def preprocess_fcs_for_model(file_path, cofactor=150.0, return_tensor=True):
    df, meta = load_fcs(file_path)
    df_fluoro = detect_fluorescence_channels(df, meta)
    df_transformed = arcsinh_transform(df_fluoro, cofactor)
    df_normalized = normalize(df_transformed)
    if return_tensor:
        return torch.tensor(df_normalized.values, dtype=torch.float32)
    else:
        return df_normalized
