import os
import csv
import spacy
import numpy as np
import streamlit as st
from io import StringIO

# Set page config
st.set_page_config(
    page_title="PLDA - POS Lexical Diversity Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to load the spaCy model, download if missing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.warning("Downloading the English language model (en_core_web_sm)...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# POS tag categories
pos_categories = {
    'Verb': 'VERB',
    'Noun': 'NOUN',
    'Adjective': 'ADJ',
    'Adverb': 'ADV'
}

# Extract words by POS
def extract_pos(text, pos_tag):
    try:
        doc = nlp(text)
        return [token.text.lower() for token in doc if token.pos_ == pos_tag]
    except Exception as e:
        st.error(f"spaCy processing error: {str(e)}")
        return []

# Calculate MATTR for a specific set of words
def calculate_mattr(words, window_size=11):
    if len(words) < window_size:
        return len(set(words)) / len(words) if words else 0
    return np.mean([len(set(words[i:i+window_size])) / window_size for i in range(len(words) - window_size + 1)])

# Calculate MATTR for a specific category (e.g., verbs, nouns) relative to all words tokens
def calculate_category_mattr(category_words, all_words, window_size=11):
    if len(category_words) < window_size or len(all_words) < window_size:
        return len(set(category_words)) / len(all_words) if all_words else 0
    return np.mean([len(set(category_words[i:i+window_size])) / window_size for i in range(len(category_words) - window_size + 1)])

# Safe tokenization function with error handling
def safe_tokenize(text):
    try:
        doc = nlp(text)
        return [token.text.lower() for token in doc if token.is_alpha]
    except Exception as e:
        st.error(f"spaCy tokenization error: {str(e)}")
        return []

# App title and description
st.title("PLDA - POS Lexical Diversity Analyzer")

with st.expander("About MATTR"):
    st.write("""
    **Moving-Average Type-Token Ratio (MATTR)** is a measure of lexical diversity.
    
    Unlike the traditional Type-Token Ratio (TTR), which is sensitive to text length, 
    MATTR calculates the average TTR over multiple smaller windows of text.
    
    This provides a more stable and comparable index of lexical diversity across samples of different lengths.
    """)

# Sidebar for options
st.sidebar.header("Analysis Options")

# Category selection checkboxes
st.sidebar.subheader("Select Categories to Analyze")
select_all = st.sidebar.checkbox("Select/Deselect All", value=True)

# Initialize checkbox states
if select_all:
    all_words = st.sidebar.checkbox("All words", value=True)
    verb = st.sidebar.checkbox("Verb", value=True)
    noun = st.sidebar.checkbox("Noun", value=True)
    adjective = st.sidebar.checkbox("Adjective", value=True)
    adverb = st.sidebar.checkbox("Adverb", value=True)
else:
    all_words = st.sidebar.checkbox("All words", value=False)
    verb = st.sidebar.checkbox("Verb", value=False)
    noun = st.sidebar.checkbox("Noun", value=False)
    adjective = st.sidebar.checkbox("Adjective", value=False)
    adverb = st.sidebar.checkbox("Adverb", value=False)

# Window size for MATTR calculation
window_size = st.sidebar.slider("MATTR Window Size", 5, 50, 11)

# File upload section
st.header("Upload Files")
uploaded_files = st.file_uploader("Upload one or more .txt files", type="txt", accept_multiple_files=True)

if uploaded_files:
    if st.button("Run Analysis"):
        selected_categories = []
        if all_words:
            selected_categories.append('All words')
        if verb:
            selected_categories.append('Verb')
        if noun:
            selected_categories.append('Noun')
        if adjective:
            selected_categories.append('Adjective')
        if adverb:
            selected_categories.append('Adverb')
        
        if not selected_categories:
            st.error("Please select at least one category to analyze.")
        else:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Prepare results
            results = []
            header = ['File Name']
            if 'All words' in selected_categories:
                header.extend(['All Words Types', 'All Words Tokens', 'All Words MATTR'])
            for pos in pos_categories:
                if pos in selected_categories:
                    header.extend([f'{pos} Types', f'{pos} Tokens', f'{pos} MATTR', f'{pos} MATTR (All Words)'])
            results.append(header)
            
            for i, file in enumerate(uploaded_files):
                try:
                    file_content = file.read().decode('utf-8')
                    status_text.text(f"Processing: {file.name}")
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    row = [file.name]
                    all_words_tokens = None
                    
                    # Option 1: All Words MATTR calculation
                    if 'All words' in selected_categories:
                        all_words_tokens = safe_tokenize(file_content)
                        types = len(set(all_words_tokens))
                        tokens = len(all_words_tokens)
                        mattr = calculate_mattr(all_words_tokens, window_size)
                        row.extend([types, tokens, round(mattr, 4)])
                    else:
                        # Still need to calculate all_words for category MATTRs
                        all_words_tokens = safe_tokenize(file_content)
                    
                    for pos in pos_categories:
                        if pos in selected_categories:
                            words = extract_pos(file_content, pos_categories[pos])
                            types = len(set(words))
                            tokens = len(words)
                            
                            # MATTR for the category
                            mattr = calculate_mattr(words, window_size)
                            
                            # MATTR for the category using all words tokens
                            mattr_category = calculate_category_mattr(words, all_words_tokens, window_size)
                            
                            row.extend([types, tokens, round(mattr, 4), round(mattr_category, 4)])
                    
                    results.append(row)
                except Exception as e:
                    st.error(f"Error processing file {file.name}: {str(e)}")
                    continue
            
            # Display results
            status_text.text("Analysis Complete!")
            progress_bar.progress(100)
            
            st.header("Analysis Results")
            st.dataframe(results[1:], columns=results[0])
            
            # Create CSV for download
            csv_data = StringIO()
            csv_writer = csv.writer(csv_data)
            for row in results:
                csv_writer.writerow(row)
            st.download_button(
                label="Download Results as CSV",
                data=csv_data.getvalue(),
                file_name="results.csv",
                mime="text/csv"
            )
