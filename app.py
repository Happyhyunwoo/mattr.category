import streamlit as st
import os
import zipfile
import tempfile
import csv
import nltk
import numpy as np
from nltk import word_tokenize, pos_tag

# Set up NLTK in a Streamlit-friendly way
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)
nltk.data.path.append(nltk_data_path)

st.title("POS-based MATTR Calculator")
st.markdown("Upload a **ZIP file** containing `.txt` files. The tool will compute POS-specific MATTR scores for each file.")

uploaded_zip = st.file_uploader("Upload a ZIP file of `.txt` files", type="zip")

def extract_pos(text, pos_prefix):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    words = [word.lower() for word, tag in tagged if tag.startswith(pos_prefix)]
    return words

def calculate_mattr(words, window_size=11):
    if len(words) < window_size:
        return len(set(words)) / len(words) if words else 0
    ratios = []
    for i in range(len(words) - window_size + 1):
        window = words[i:i + window_size]
        ratio = len(set(window)) / window_size
        ratios.append(ratio)
    return np.mean(ratios)

if uploaded_zip:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        results_path = os.path.join(tmpdir, "results.csv")

        pos_categories = {
            'Verb': 'VB',
            'Noun': 'NN',
            'Adjective': 'JJ',
            'Adverb': 'RB'
        }

        with open(results_path, 'w', newline='', encoding='utf-8') as results_file:
            csv_writer = csv.writer(results_file)
            header = ['File Name']
            for pos in pos_categories:
                header.extend([f'{pos} Types', f'{pos} Tokens', f'{pos} MATTR'])
            csv_writer.writerow(header)

            for filename in os.listdir(tmpdir):
                if filename.endswith('.txt'):
                    file_path = os.path.join(tmpdir, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    row = [filename]
                    for pos, prefix in pos_categories.items():
                        words = extract_pos(content, prefix)
                        types = len(set(words))
                        tokens = len(words)
                        mattr = calculate_mattr(words)
                        row.extend([types, tokens, f"{mattr:.4f}"])
                    csv_writer.writerow(row)

        with open(results_path, "rb") as f:
            st.download_button(
                label="Download Results as CSV",
                data=f,
                file_name="results.csv",
                mime="text/csv"
            )
