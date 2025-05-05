import streamlit as st
import os
import tempfile
import csv
import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize, pos_tag
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="POS-based MATTR Calculator",
    page_icon="📊",
    layout="wide"
)

# Create sidebar for app information
with st.sidebar:
    st.title("About this App")
    st.markdown("""
    This application calculates Moving Average Type-Token Ratio (MATTR) 
    by Part of Speech (POS) categories for text files.
    
    **Key Features:**
    - Upload multiple text files at once
    - Customize window size for MATTR calculation
    - See results as a table
    - Download results as CSV
    
    **POS Categories:**
    - Verbs (VB*)
    - Nouns (NN*)
    - Adjectives (JJ*)
    - Adverbs (RB*)
    """)

# Main app title and description
st.title("📊 POS-based MATTR Calculator")
st.markdown("""
Upload multiple `.txt` files to calculate Moving Average Type-Token Ratio (MATTR) 
per Part of Speech category for each file.
""")

# Interface for MATTR window size configuration
window_size = st.number_input(
    "MATTR Window Size", 
    min_value=5, 
    max_value=100, 
    value=11,
    help="The size of the moving window to calculate MATTR. Recommended values range from 10-100."
)

# File uploader
uploaded_files = st.file_uploader(
    "Upload `.txt` files", 
    type="txt", 
    accept_multiple_files=True,
    help="Select multiple text files to analyze"
)

# Setup NLTK data path
@st.cache_resource
def setup_nltk():
    nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
    os.makedirs(nltk_data_path, exist_ok=True)
    nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path, quiet=True)
    nltk.data.path.append(nltk_data_path)
    return True

# Extract words with specific POS tag prefix
def extract_pos(text, pos_prefix):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    words = [word.lower() for word, tag in tagged if tag.startswith(pos_prefix)]
    return words

# Calculate MATTR for a list of words
def calculate_mattr(words, window_size=11):
    if len(words) < window_size:
        return len(set(words)) / len(words) if words else 0
    
    ratios = []
    for i in range(len(words) - window_size + 1):
        window = words[i:i + window_size]
        ratio = len(set(window)) / window_size
        ratios.append(ratio)
    
    return np.mean(ratios) if ratios else 0

# Initialize NLTK
setup_nltk()

# Define POS categories
pos_categories = {
    'Verb': 'VB',
    'Noun': 'NN',
    'Adjective': 'JJ',
    'Adverb': 'RB'
}

if uploaded_files:
    st.write(f"Processing {len(uploaded_files)} file(s)...")
    
    # Progress bar
    progress_bar = st.progress(0)
    
    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as tmpdir:
        results_path = os.path.join(tmpdir, "results.csv")
        
        # Prepare CSV header
        header = ['File Name']
        for pos in pos_categories:
            header.extend([f'{pos} Types', f'{pos} Tokens', f'{pos} MATTR'])
        
        # Process each file
        results = []
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Read file content
                content = uploaded_file.read().decode('utf-8', errors='replace')
                
                # Process file
                row = [uploaded_file.name]
                file_data = {}
                
                for pos, prefix in pos_categories.items():
                    words = extract_pos(content, prefix)
                    types = len(set(words))
                    tokens = len(words)
                    mattr = calculate_mattr(words, window_size=window_size)
                    
                    row.extend([types, tokens, f"{mattr:.4f}"])
                    
                    # Store data for display
                    file_data[f'{pos} Types'] = types
                    file_data[f'{pos} Tokens'] = tokens
                    file_data[f'{pos} MATTR'] = f"{mattr:.4f}"
                
                results.append([uploaded_file.name, file_data])
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        # Create CSV file with results
        csv_data = StringIO()
        csv_writer = csv.writer(csv_data)
        csv_writer.writerow(header)
        
        for filename, data in results:
            row = [filename]
            for pos in pos_categories:
                row.extend([
                    data[f'{pos} Types'], 
                    data[f'{pos} Tokens'], 
                    data[f'{pos} MATTR']
                ])
            csv_writer.writerow(row)
        
        # Complete progress bar
        progress_bar.progress(1.0)
        
        # Display results
        st.subheader("Results")
        
        # Convert results to DataFrame for display
        df_data = {
            'File Name': [r[0] for r in results]
        }
        
        for pos in pos_categories:
            df_data[f'{pos} Types'] = [r[1][f'{pos} Types'] for r in results]
            df_data[f'{pos} Tokens'] = [r[1][f'{pos} Tokens'] for r in results]
            df_data[f'{pos} MATTR'] = [r[1][f'{pos} MATTR'] for r in results]
        
        results_df = pd.DataFrame(df_data)
        st.dataframe(results_df, use_container_width=True)
        
        # Download button
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data.getvalue(),
            file_name="pos_mattr_results.csv",
            mime="text/csv"
        )

else:
    st.info("Upload text files to begin analysis.")
    
    # Example section
    with st.expander("See an example of the output"):
        example_df = pd.DataFrame({
            'File Name': ['example1.txt', 'example2.txt'],
            'Verb Types': [120, 95],
            'Verb Tokens': [350, 280],
            'Verb MATTR': ['0.6523', '0.5978'],
            'Noun Types': [180, 150],
            'Noun Tokens': [400, 320],
            'Noun MATTR': ['0.7123', '0.6854'],
            'Adjective Types': [75, 60],
            'Adjective Tokens': [150, 130],
            'Adjective MATTR': ['0.6012', '0.5789'],
            'Adverb Types': [45, 35],
            'Adverb Tokens': [100, 85],
            'Adverb MATTR': ['0.5567', '0.5321']
        })
        st.dataframe(example_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("*MATTR = Moving Average Type-Token Ratio*")
