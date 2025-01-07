from markitdown import MarkItDown # type: ignore
import spacy
import paths
import os
import pickle
import tqdm


md = MarkItDown()
nlp = spacy.load("en_core_web_lg")

# to correct the accent marks
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




# to get the text from file
def get_text(path):
    result = md.convert(path)
    return correct_accent(result.text_content)





# to get the chunks from the text
def get_chunks(text, chunk_size=1000, overlap=200):
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # Move the start position forward, keeping the overlap
        start += chunk_size - overlap

    return chunks



# Traverse through all PDF files in a directory
def get_all_files(directory):
    """Returns a list of all files in the given directory, including files in subdirectories."""
    files_list = []
    file2index = {}

    # Traverse the directory and its subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            # Create full path and append to the list
            file_path = os.path.join(root, file)
            files_list.append(file_path)
            file2index[file_path] = len(files_list) - 1

    return files_list, file2index




# tokenize text for bm25 model with spacy
def tokenize(text):
    """Tokenizes all the texts in the dictionary."""
    max_length = 1000000  # Maximum length for spaCy
    tokens = []
    for i in range(0, len(text), max_length):
        doc = nlp(text[i:i + max_length])
        tokens += [token.text.lower() for token in doc if not token.is_stop]
    return tokens


# function to save an object with pickle
def save_object(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

# function to load an object with pickle
def load_object(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# get the chunks from the directory
def get_chunks_from_directory(directory):
    files, file2index = get_all_files(directory)
    chunks = []
    chunk2info = {}

    for file in tqdm.tqdm(files, desc="Processing files for chunks"):
        text = get_text(file)
        file_chunks = get_chunks(text)  # Récupérer les chunks pour ce fichier
        for chunk in file_chunks:
            chunks.append(chunk)
            chunk2info[len(chunks) - 1] = (chunk, file, file2index[file])

    return chunks, chunk2info

# get the tokens from the chunks
def get_tokens_from_chunks(chunks, chunk2info):
    #chunks, chunk2info = get_chunks_from_directory(directory)
    tokens = []
    chunk2file = {}

    for chunk_index, chunk in tqdm.tqdm(enumerate(chunks), total=len(chunks), desc="Processing chunks for tokens"):
        tokens.append(tokenize(chunk))  # Tokeniser le chunk
        _, file, file_index = chunk2info[chunk_index]
        chunk2file[chunk_index] = (chunk, file, file_index)

    return tokens, chunk2file

# save the data to files
def save_data(chunks, chunk2file, tokens, path):
    save_object(chunks, os.path.join(path, "chunks"))
    save_object(chunk2file, os.path.join(path, "chunk2file"))
    save_object(tokens, os.path.join(path, "tokens"))

# load the data from files
def load_data(path):
    chunks = load_object(os.path.join(path, "chunks"))
    chunk2file = load_object(os.path.join(path, "chunk2file"))
    tokens = load_object(os.path.join(path, "tokens"))
    return chunks, chunk2file, tokens

"""# path to the PDF files directory
directory_path = paths.data_path

# get the chunks from the directory
print("Getting chunks from directory...")
chunks, chunk2info = get_chunks_from_directory(directory_path)

# get the tokens from the chunks
print("Getting tokens from chunks...")
tokens, chunk2file = get_tokens_from_chunks(chunks, chunk2info)

# verify the path for the preprocessed data
saved_path = paths.preprocessed_data2
os.makedirs(saved_path, exist_ok=True)

# save the data to files
print(f"Saving data to {saved_path}...")
save_data(chunks, chunk2file, tokens, saved_path)

print("Preprocessing completed!")"""
