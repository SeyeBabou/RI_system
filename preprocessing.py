import os
import fitz
import spacy
import paths
import numpy as np
from collections import defaultdict
import pickle

# Load the English language model
nlp = spacy.load("en_core_web_lg")

def correct_accent(text):
    """Corrects accent marks."""
    
    # Create a new list to hold the corrected characters
    corrected_text = []
    i = 0

    while i < len(text):
        if text[i] == '´':
            if i + 1 < len(text) and text[i + 1] == 'e':
                corrected_text.append('é')
                i += 2  # Skip the next character
                continue
        elif text[i] == '`':
            if i + 1 < len(text):
                if text[i + 1] == 'e':
                    corrected_text.append('è')
                elif text[i + 1] == 'a':
                    corrected_text.append('à')
                elif text[i + 1] == 'u':
                    corrected_text.append('ù')
                i += 2  # Skip the next character
                continue

        # If no replacement is done, just append the current character
        corrected_text.append(text[i])
        i += 1

    # Join the list into a string and return it
    return ''.join(corrected_text)

# Traverse through all PDF files in a directory
def get_all_files(directory):
    """Returns a list of all files in the given directory, including files in subdirectories."""
    files_list = []

    # Traverse the directory and its subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            # Create full path and append to the list
            files_list.append(os.path.join(root, file))

    return files_list

def tokenize(text):
    """Tokenizes all the texts in the dictionary."""
    max_length = 1000000  # Maximum length for spaCy
    tokens = []
    for i in range(0, len(text), max_length):
        doc = nlp(text[i:i + max_length])
        tokens += [token.text.lower() for token in doc if not token.is_stop]
    return tokens

def preprocess_data(directory):
    """Load the data -> pdf_file: text."""
    
    # Function to extract text from a PDF file
    def extract_text_from_pdf(pdf_path):
        try:
            with fitz.open(pdf_path) as doc:
                extracted_text = ''
                for page in doc:
                    extracted_text += page.get_text() + '\n'
        except Exception as e:
            print(f"Error opening the file {pdf_path}: {e}")
            return ""

        utf8_text = extracted_text.encode('utf-8')
        return correct_accent(utf8_text.decode('utf-8'))

    all_files = get_all_files(directory)
    index2tokens = {}
    index2file = {}
    index2text = {}
    index = 0
    for file in all_files:
        print(f"{file}")
        text = extract_text_from_pdf(file)
        tokens = tokenize(text)
        filename = file.split('/')[-1].split('.')[0]
        index2tokens[index] = tokens
        index2file[index] = file
        index2text[index] = text
        index += 1
    return index2tokens , index2file , index2text

def create_inverted_index(directory):
    """Create the inverted index."""
    ind2tok , ind2file , ind2text = preprocess_data(directory)
    inverted_index = defaultdict(list)
    for doc_id, words in ind2tok.items():
        for word in set(words):  # Use a set to avoid duplicates
            inverted_index[word].append(doc_id)
    return inverted_index , ind2tok , ind2file , ind2text

def save_data(inverted_index, ind2tok, ind2file, ind2text, prefix):
    """Save the data to files."""
    with open(f'{prefix}_inverted_index.pkl', 'wb') as f:
        pickle.dump(inverted_index, f)
    with open(f'{prefix}_ind2tok.pkl', 'wb') as f:
        pickle.dump(ind2tok, f)
    with open(f'{prefix}_ind2file.pkl', 'wb') as f:
        pickle.dump(ind2file, f)
    with open(f'{prefix}_ind2text.pkl', 'wb') as f:
        pickle.dump(ind2text, f)

def load_data(prefix):
    """Load data from files."""
    with open(f'{prefix}_inverted_index.pkl', 'rb') as f:
        inverted_index = pickle.load(f)
    with open(f'{prefix}_ind2tok.pkl', 'rb') as f:
        ind2tok = pickle.load(f)
    with open(f'{prefix}_ind2file.pkl', 'rb') as f:
        ind2file = pickle.load(f)
    with open(f'{prefix}_ind2text.pkl', 'rb') as f:
        ind2text = pickle.load(f)
    
    return inverted_index, ind2tok, ind2file, ind2text

preprocessed_data_path = str(paths.preprocessed_data)

#inverted_index, ind2tok, ind2file, ind2text = create_inverted_index(paths.data_path)
#save_data(inverted_index, ind2tok, ind2file, ind2text, "preprocessed_data")

#print(f"The data has been saved in the folder {preprocessed_data_path}")
