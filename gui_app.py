import streamlit as st
import torch
from model import FlexibleCompensator
from fcs_preprocessing import preprocess_fcs_for_model
from visualize import plot_umap
import pandas as pd
import zipfile
import io
import os
import json

st.title("Flow Compensation Model")

uploaded_files = st.file_uploader("Upload one or more raw `.fcs` files", type=["fcs"], accept_multiple_files=True)

if uploaded_files:
    compensated_csvs = []
    umap_data = []
    for uploaded_file in uploaded_files:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        raw_df = preprocess_fcs_for_model(temp_path, return_tensor=False)
        # Align columns to match training
        # Use the same column cleaning and sorting as in training
        def clean_col(col):
            return col.lower().replace('-a', '').replace('-comp', '').replace(' ', '').replace('_', '')
        # Load the trained model's input and output columns
        if os.path.exists("model_columns.json"):
            with open("model_columns.json", "r") as f:
                model_cols = json.load(f)
            not_unmixed_cols = model_cols["not_unmixed_cols"]
            unmixed_cols = model_cols["unmixed_cols"]
        else:
            # Fallback: use columns from the uploaded file
            not_unmixed_cols = sorted([col for col in raw_df.columns if 'spectral' not in col.lower()], key=clean_col)
            unmixed_cols = sorted([col for col in raw_df.columns if 'spectral' in col.lower()], key=clean_col)
        # Use only the columns that match the trained model
        missing_cols = [col for col in not_unmixed_cols if col not in raw_df.columns]
        if missing_cols:
            st.error(f"File {uploaded_file.name} is missing required columns for the model: {missing_cols}. Skipping this file.")
            os.remove(temp_path)
            continue
        input_tensor = torch.tensor(raw_df[not_unmixed_cols].values, dtype=torch.float32)

        model = FlexibleCompensator(len(not_unmixed_cols), [1024, 2048, 1024, 512], len(unmixed_cols), activation="SiLU", norm="LayerNorm", dropout=0.1)
        model.load_state_dict(torch.load("trained_model.pt", map_location="cpu"))
        model.eval()
        with torch.no_grad():
            comp = model(input_tensor)

        comp_df = pd.DataFrame(comp.numpy(), columns=unmixed_cols)
        compensated_csvs.append((uploaded_file.name.replace('.fcs', '_compensated.csv'), comp_df))
        umap_data.append((raw_df, comp_df, uploaded_file.name))
        os.remove(temp_path)

    # Create a zip of all compensated CSVs
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for fname, df in compensated_csvs:
            csv_bytes = df.to_csv(index=False).encode()
            zipf.writestr(fname, csv_bytes)
    st.success(f"Compensation applied to {len(compensated_csvs)} files.")
    st.download_button("Download All Compensated CSVs (zip)", zip_buffer.getvalue(), file_name="compensated_csvs.zip")

    st.subheader("UMAP Plots: Before vs After Compensation")
    for raw_df, comp_df, fname in umap_data:
        st.markdown(f"**{fname}**")
        plot_umap(raw_df, comp_df)
