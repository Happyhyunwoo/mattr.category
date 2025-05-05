#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import csv
import nltk
from nltk import word_tokenize, pos_tag
import numpy as np
import streamlit as st
from io import StringIO

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Function to extract POS tags
def extract_pos(text, pos_prefix):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    words = [word.lower() for word, tag in tagged if tag.startswith(pos_prefix)]
    return words

# Function to calculate MATTR (Moving Average Type-Token Ratio)
def calculate_mattr(words, window_size=11):
    if len(words) < window_size:
        return len(set(words)) / len(words) if words else 0
    
    ratios = []
    for i in range(len(words) - window_size + 1):
        window = words[i:i+window_size]
        ratio = len(set(window)) / window_size  # Number of unique words in the window divided by the window size
        ratios.append(ratio)
    
    return np.mean(ratios)  # Average of all the ratios

# Streamlit UI components
st.title("Text File Analysis Tool")

# Upload a folder of text files
uploaded_files = st.file_uploader("Upload Text Files", type=["txt"], accept_multiple_files=True)

# POS categories
pos_categories = {
    'Verb': 'VB',
    'Noun': 'NN',
    'Adjective': 'JJ',
    'Adverb': 'RB'
}

if uploaded_files:
    # Create a StringIO object to hold the CSV data
    output = StringIO()
    csv_writer = csv.writer(output)

    # Write the header
    header = ['File Name']
    for pos in pos_categories:
        header.extend([f'{pos} Types', f'{pos} Tokens', f'{pos} MATTR'])
    csv_writer.writerow(header)
    
    # Process each uploaded file
    for uploaded_file in uploaded_files:
        content = uploaded_file.getvalue().decode("utf-8")
        filename = uploaded_file.name

        # Process each POS category for the current file
        row = [filename]
        for pos, prefix in pos_categories.items():
            words = extract_pos(content, prefix)
            types = len(set(words))  # Count unique types (distinct words)
            tokens = len(words)  # Count total tokens (all words)
            mattr = calculate_mattr(words)  # Calculate MATTR
            row.extend([types, tokens, f"{mattr:.4f}"])  # Append the results to the row
        
        # Write the data row to the CSV output
        csv_writer.writerow(row)
    
    # Provide the user with an option to download the result CSV file
    output.seek(0)  # Reset the StringIO object to the beginning
    st.download_button(
        label="Download Results CSV",
        data=output.getvalue(),
        file_name="results.csv",
        mime="text/csv"
    )

    st.success("Analysis Complete! You can download the results.")

