import streamlit as st
import os
import tempfile
import csv
import numpy as np
import pandas as pd
import re
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="POS-based MATTR Calculator",
    page_icon="📊",
    layout="wide"
)

# Main app title and description
st.title("📊 POS-based MATTR Calculator")
st.markdown("""
Upload multiple `.txt` files to calculate Moving Average Type-Token Ratio (MATTR) 
per Part of Speech category for each file.
""")

# Create sidebar for app information
with st.sidebar:
    st.title("About this App")
    st.markdown("""
    This application calculates Moving Average Type-Token Ratio (MATTR) 
    by Part of Speech (POS) categories for text files.
    
    **Key Features:**
    - Upload multiple text files at once
    - Fixed window size of 11 for MATTR calculation
    - See results as a table
    - Download results as CSV
    
    **POS Categories:**
    - Verbs (VB*)
    - Nouns (NN*)
    - Adjectives (JJ*)
    - Adverbs (RB*)
    """)

# Create a toggle for using simple tokenization
use_simple_processing = st.checkbox(
    "Use simple tokenization (recommended if NLTK errors occur)",
    value=True,
    help="Enable this option if you encounter NLTK-related errors"
)

# File uploader
uploaded_files = st.file_uploader(
    "Upload `.txt` files", 
    type="txt", 
    accept_multiple_files=True,
    help="Select multiple text files to analyze"
)

# Simple tokenization and POS tagging functions that don't rely on NLTK
def simple_tokenize(text):
    """Tokenize text using simple regex patterns."""
    # Convert to lowercase and strip punctuation
    text = text.lower()
    # Split on whitespace and remove empty strings
    tokens = re.findall(r'\b[a-z0-9]+\b', text)
    return tokens

# Simple rule-based POS tagging
def simple_pos_categorize(token):
    """Categorize token into basic POS categories using simple rules."""
    # Common verb endings
    if token.endswith(('ed', 'ing', 'ize', 'ise', 'ate', 'en')) or token in [
        'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'go', 'goes', 'went',
        'see', 'saw', 'seen', 'come', 'came', 'run', 'ran', 'get', 'got',
        'make', 'made', 'take', 'took', 'give', 'gave', 'live', 'say',
        'said', 'tell', 'told', 'find', 'found', 'think', 'thought',
        'know', 'knew', 'want', 'put', 'read', 'write', 'wrote'
    ]:
        return 'VB'  # Verb
    
    # Common adjective endings
    elif token.endswith(('able', 'ible', 'al', 'ful', 'ic', 'ive', 'less', 'ous')) or token in [
        'good', 'bad', 'new', 'old', 'high', 'low', 'big', 'small', 'large',
        'little', 'long', 'short', 'great', 'best', 'better', 'worst', 'worse',
        'nice', 'fine', 'happy', 'sad', 'hot', 'cold', 'warm', 'cool', 'slow',
        'fast', 'easy', 'hard', 'early', 'late', 'young', 'red', 'blue', 'green',
        'black', 'white', 'dark', 'light', 'rich', 'poor', 'real', 'true', 'false'
    ]:
        return 'JJ'  # Adjective
    
    # Common adverb endings
    elif token.endswith('ly') or token in [
        'very', 'too', 'so', 'quite', 'well', 'now', 'then', 'here', 'there',
        'just', 'only', 'also', 'even', 'still', 'again', 'already', 'always',
        'never', 'often', 'sometimes', 'usually', 'today', 'tomorrow', 'yesterday',
        'soon', 'later', 'early', 'fast', 'hard', 'away', 'back', 'up', 'down',
        'in', 'out', 'off', 'on', 'over', 'under'
    ]:
        return 'RB'  # Adverb
    
    # Common noun endings
    elif token.endswith(('tion', 'sion', 'ment', 'ness', 'ity', 'ship', 'dom', 'er', 'or', 'ian', 'ist')) or token in [
        'time', 'year', 'day', 'man', 'woman', 'child', 'person', 'world', 'life',
        'hand', 'part', 'place', 'case', 'group', 'company', 'party', 'school',
        'country', 'state', 'family', 'money', 'night', 'water', 'thing', 'name',
        'book', 'room', 'area', 'point', 'house', 'home', 'job', 'line', 'end',
        'city', 'car', 'team', 'word', 'game', 'food', 'paper', 'music', 'problem'
    ]:
        return 'NN'  # Noun
    
    # Default case - use most common POS category: nouns
    else:
        # Most unknown words are likely nouns
        return 'NN'  # Default to noun

# Extract words with specific POS tag prefix (using simple categorization)
def extract_pos_simple(text, pos_prefix):
    """Extract words with a specific POS tag prefix using simple rules."""
    try:
        if not isinstance(text, str):
            text = str(text)
        
        tokens = simple_tokenize(text)
        tagged = [(token, simple_pos_categorize(token)) for token in tokens]
        words = [word for word, tag in tagged if tag.startswith(pos_prefix)]
        return words
    except Exception as e:
        st.error(f"Error in extract_pos_simple: {str(e)}")
        return []

# Try to import and configure NLTK if advanced processing is requested
if not use_simple_processing:
    try:
        # Use try/except to handle NLTK import
        with st.spinner("Loading NLTK resources..."):
            import nltk
            from nltk import word_tokenize, pos_tag
            
            # Set up NLTK data path
            nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
            os.makedirs(nltk_data_path, exist_ok=True)
            
            # Configure NLTK data path
            nltk.data.path.insert(0, nltk_data_path)
            
            # Download required resources with proper error handling
            try:
                nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
                nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path, quiet=True)
                st.success("NLTK resources loaded successfully")
                
                # Test tokenization
                test_text = "This is a test sentence."
                test_tokens = word_tokenize(test_text)
                test_tags = pos_tag(test_tokens)
                
                if len(test_tags) > 0:
                    st.success("NLTK tokenization and tagging is working")
                else:
                    st.warning("NLTK tokenization test failed, falling back to simple processing")
                    use_simple_processing = True
                
            except Exception as e:
                st.warning(f"Failed to download NLTK resources: {str(e)}")
                st.info("Falling back to simple processing mode")
                use_simple_processing = True
                
            # Define the advanced POS extraction function
            def extract_pos_nltk(text, pos_prefix):
                """Extract words with specific POS tag prefix using NLTK."""
                try:
                    if not isinstance(text, str):
                        text = str(text)
                    
                    tokens = word_tokenize(text)
                    tagged = pos_tag(tokens)
                    words = [word.lower() for word, tag in tagged if tag.startswith(pos_prefix)]
                    return words
                except Exception as e:
                    st.warning(f"NLTK processing error: {str(e)}")
                    # Fall back to simple processing if NLTK fails
                    return extract_pos_simple(text, pos_prefix)
                
    except ImportError:
        st.warning("NLTK import failed. Using simple processing mode instead.")
        use_simple_processing = True

# Calculate MATTR for a list of words with fixed window size of 11
def calculate_mattr(words):
    """Calculate Moving Average Type-Token Ratio with fixed window size of 11."""
    if not words:
        return 0
        
    # Fixed window size of 11
    window_size = 11
        
    if len(words) < window_size:
        return len(set(words)) / len(words) if words else 0
    
    ratios = []
    for i in range(len(words) - window_size + 1):
        window = words[i:i + window_size]
        ratio = len(set(window)) / window_size
        ratios.append(ratio)
    
    return np.mean(ratios) if ratios else 0

# Define POS categories
pos_categories = {
    'Verb': 'VB',
    'Noun': 'NN',
    'Adjective': 'JJ',
    'Adverb': 'RB'
}

# Select the appropriate POS extraction function based on user choice
extract_pos = extract_pos_simple if use_simple_processing else extract_pos_nltk

# Processing indication
if use_simple_processing:
    st.info("Using simple rule-based tokenization and POS tagging")
else:
    st.info("Using NLTK for tokenization and POS tagging")

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
                
                # Read file content with robust error handling
                try:
                    content = uploaded_file.read().decode('utf-8', errors='replace')
                except Exception as read_error:
                    st.warning(f"Error reading {uploaded_file.name}: {str(read_error)}")
                    st.info("Attempting alternative reading method...")
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        content = ""
                        for line in uploaded_file:
                            content += line.decode('utf-8', errors='replace')
                    except Exception as alt_error:
                        st.error(f"All reading attempts failed for {uploaded_file.name}")
                        content = ""  # Use empty content to continue processing
                
                # Process file
                row = [uploaded_file.name]
                file_data = {}
                
                for pos, prefix in pos_categories.items():
                    words = extract_pos(content, prefix)
                    types = len(set(words))
                    tokens = len(words)
                    mattr = calculate_mattr(words)
                    
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
