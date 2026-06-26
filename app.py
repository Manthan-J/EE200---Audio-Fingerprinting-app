import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import sqlite3
import shutil
from pathlib import Path
import pandas as pd
from huggingface_hub import snapshot_download, HfApi

# -------------------------------------------------------------------------
# 1. HUGGING FACE DATASET PERSISTENCE CONFIGURATION
# -------------------------------------------------------------------------
# Replace with your actual repository details
REPO_ID = "Manthan-J/EE200_Project_Song_Database" 

# Read the secure Write Token from Space Secrets
HF_TOKEN = os.environ.get("HF_TOKEN")

# Define local persistent working directory paths inside the Space
target_folder = "working_audio_folder"
db_name_default = "working_database.db"

# Initialize HfApi client if token is available
api = HfApi(token=HF_TOKEN) if HF_TOKEN else None

@st.cache_resource
def initialize_data_from_hf():
    """Downloads baseline data from the HF Dataset on initial startup."""
    print("Downloading baseline data from Hugging Face Dataset...")
    dataset_dir = snapshot_download(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
    
    # Copy the DB to writable workspace
    cached_db_path = os.path.join(dataset_dir, "song_database.db")
    if os.path.exists(cached_db_path):
        shutil.copy2(cached_db_path, db_name_default)
    
    # Copy the audio files to writable workspace
    # Check if your dataset has a subfolder, otherwise fall back to top-level directory
    cached_audio_folder = os.path.join(dataset_dir, "EE200 Project Song Database")
    if os.path.exists(cached_audio_folder):
        shutil.copytree(cached_audio_folder, target_folder, dirs_exist_ok=True)
    else:
        # If songs are loose in the main repository directory
        os.makedirs(target_folder, exist_ok=True)
        for item in os.listdir(dataset_dir):
            item_path = os.path.join(dataset_dir, item)
            if os.path.isfile(item_path) and item.endswith(('.mp3', '.wav', '.flac')):
                shutil.copy2(item_path, os.path.join(target_folder, item))

# Run the initialization on startup
initialize_data_from_hf()


# -------------------------------------------------------------------------
# 2. APPLICATION IMPORTS & CONFIGURATION
# -------------------------------------------------------------------------
from fingerprint import (get_peaks, process_song_database, match_audio_clip, 
                         plot_offset_histogram, plot_constellation_of_peaks, 
                         plot_spectrogram, store_song_fingerprints)

st.set_page_config(page_title="EE200: Audio Fingerprinting", layout="wide", initial_sidebar_state="collapsed")
st.title("Zapptain America: Audio Fingerprinting")
st.markdown("#SIGNALS, SYSTEMS & NETWORKS Project")
st.markdown("This app is a mini version of the famous Shazam App. It takes a song from the user, finds its constellation peaks, matches them with a song present in our database, and gives its name.")

# Create the top-level navigation tabs
tab_lib, tab_identify, tab_batch = st.tabs([" LIBRARY", " IDENTIFY", " BATCH"])

# -------------------------------------------------------------------------
# 3. LIBRARY MANAGEMENT (WITH PERSISTENT UPLOADS)
# -------------------------------------------------------------------------
with tab_lib:
    st.markdown("### Library Management")
    st.markdown("####  Add a New Song :-")
    st.markdown("Upload a full track to add it permanently to the database.")
    uploaded_lib_file = st.file_uploader("Upload a song (.wav, .mp3, .flac)", type=["wav", "mp3", "flac"], key="lib_upload")
    
    if uploaded_lib_file is not None:
        if st.button("Index Uploaded Song", type="primary"):
            if not HF_TOKEN:
                st.error("⚠️ HF_TOKEN secret is missing in Space Settings! Cannot save permanently.")
            else:
                with st.spinner(f"Analyzing, indexing, and uploading '{uploaded_lib_file.name}'..."):
                    # A. Save the file locally to the active session workspace
                    os.makedirs(target_folder, exist_ok=True)
                    file_path = os.path.join(target_folder, uploaded_lib_file.name) 
                    with open(file_path, "wb") as f:
                        f.write(uploaded_lib_file.getbuffer())
                    
                    # B. Extract peaks and generate hashes in local database
                    song_name = uploaded_lib_file.name.rsplit('.', 1)[0]
                    WINDOW_SIZE = 4096
                    HOP_LENGTH = WINDOW_SIZE // 4
                    _, times, freqs, _ = get_peaks(file_path, WINDOW_SIZE, HOP_LENGTH)
                    
                    conn = sqlite3.connect(db_name_default)
                    num_hashes = store_song_fingerprints(conn, times, freqs, song_name)
                    conn.close()
                    
                    # C. PUSH CHANGES TO HUGGING FACE DATASET TO PREVENT WIPING OUT
                    try:
                        # 1. Upload the updated database file
                        api.upload_file(
                            path_or_fileobj=db_name_default,
                            path_in_repo="song_database.db",
                            repo_id=REPO_ID,
                            repo_type="dataset"
                        )
                        
                        # 2. Upload the newly added audio track
                        # Saves it inside a subfolder matching your layout, change path_in_repo if needed
                        api.upload_file(
                            path_or_fileobj=file_path,
                            path_in_repo=f"EE200 Project Song Database/{uploaded_lib_file.name}",
                            repo_id=REPO_ID,
                            repo_type="dataset"
                        )
                        
                        st.success(f"Successfully added '{song_name}'! Generated {num_hashes} hashes and committed to Hugging Face Dataset.")
                    except Exception as e:
                        st.error(f"Failed to sync changes to Hugging Face: {e}")

    st.divider()
    st.markdown("####  Current Database")
    
    try:
        conn = sqlite3.connect(db_name_default)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT song_id FROM fingerprints ORDER BY song_id")
        indexed_songs = [row[0] for row in cursor.fetchall()]
        conn.close()
    except sqlite3.OperationalError:
        indexed_songs = []
        
    if not indexed_songs:
        st.info("The database is currently empty. Upload a song above to get started.")
    else:
        st.metric(label="Total Songs Indexed", value=len(indexed_songs))
        selected_song = st.selectbox("Select a song to view its constellation peaks:", indexed_songs)
        if selected_song:
            folder_path = Path(target_folder)
            matching_files = list(folder_path.glob(f"{selected_song}.*"))
            if matching_files:
                target_file = matching_files[0]
                with st.spinner(f"Rendering constellation for '{selected_song}'..."):
                    WINDOW_SIZE = 4096
                    HOP_LENGTH = WINDOW_SIZE // 4
                    spectrogram_db, times, freqs, samplerate = get_peaks(str(target_file), WINDOW_SIZE, HOP_LENGTH)
                    fig = plot_constellation_of_peaks(spectrogram_db, times, freqs, HOP_LENGTH, selected_song, samplerate)
                    if fig:
                        st.pyplot(fig)
            else:
                st.warning(f"Audio file for '{selected_song}' is missing from the working folder.")

# ... [Keep your existing 'Identify a clip' and 'Identify many clips at once' code down here] ...

#Single Clip Mode
with tab_identify:
    st.markdown("### Identify a clip")
    uploaded_file = st.file_uploader("Upload a  clip", type=["wav", "mp3", "flac", "ogg", "m4a"], key="single_upload")
    # We need a default db_name for the identify and batch tabs
    if uploaded_file is not None:
        temp_clip_path = f"temp_query.{uploaded_file.name.split('.')[-1]}"
        with open(temp_clip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.audio(temp_clip_path)
        if st.button("Identify", type="primary"):
            with st.spinner("Extracting peaks and querying database..."):
                # Run matching
                matched_song, histogram = match_audio_clip(temp_clip_path, db_name_default)
                if matched_song:
                    st.success(f"**MATCH FOUND:** {matched_song}")
                    # Visualizations
                    WINDOW_SIZE = 4096
                    HOP_LENGTH = WINDOW_SIZE // 4
                    spectrogram_db, times, freqs, samplerate = get_peaks(temp_clip_path, WINDOW_SIZE, HOP_LENGTH)
                    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Spectrogram", "Constellation", "Alignment Spike"])
                    with viz_tab1:
                        fig1 = plot_spectrogram(spectrogram_db, HOP_LENGTH, samplerate, "Query Clip")
                        st.pyplot(fig1)
                        
                    with viz_tab2:
                        fig2 = plot_constellation_of_peaks(spectrogram_db, times, freqs, HOP_LENGTH, "Query Clip", samplerate)
                        st.pyplot(fig2)
                        
                    with viz_tab3:
                        fig3 = plot_offset_histogram(histogram, matched_song)
                        if fig3:
                            st.pyplot(fig3)
                        else:
                            st.info("Not enough data to plot the histogram.")
                else:
                    st.error("No match found in the database.")
                    
        # Cleanup
        if os.path.exists(temp_clip_path):
            os.remove(temp_clip_path)

#Multi Clip Mode
with tab_batch:
    st.markdown("### Identify many clips at once")
    st.markdown("Upload a set of query clips. Results are written to a standardized `results.csv`.") 
    batch_files = st.file_uploader("Upload multiple query clips", type=["wav", "mp3", "flac"], accept_multiple_files=True, key="batch_upload")
    if batch_files:
        if st.button("Run Batch", type="primary"):
            results_data = []
            progress_bar = st.progress(0)# Create progress bar
            status_text = st.empty()
            for i, file in enumerate(batch_files):
                status_text.text(f"Identifying: {file.name} ({i+1}/{len(batch_files)})")
                temp_path = f"temp_batch_{i}.{file.name.split('.')[-1]}"# Save temp file
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                matched_song, _ = match_audio_clip(temp_path, db_name_default)# Run matching
                prediction = matched_song if matched_song else "None"
                results_data.append({
                    "filename": file.name,
                    "prediction": prediction
                })
                if os.path.exists(temp_path):#deletes the temperory file
                    os.remove(temp_path)  
                progress_bar.progress((i + 1) / len(batch_files))# Update progress
            status_text.text("processing complete!")
            df = pd.DataFrame(results_data)# Generates CSV
            st.dataframe(df) # Show the table on screen
            
            # Create download button
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download the results",
                data=csv_data,
                file_name='results.csv',
                mime='text/csv',
                type="primary"
            )