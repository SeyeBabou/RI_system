import os
import fitz
import spacy
import paths
import numpy as np
from collections import defaultdict
import pickle



# Charger le modèle de langue français
nlp = spacy.load("en_core_web_lg")




def correct_accent(text):
    """corriger les accents"""
    
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




#parcourir les fichiers pdf
def get_all_files(directory):
    """Returns a list of all files in the given directory, including files in subdirectories."""
    files_list = []

    # Traverse the directory and its subdirectories
    for root, _ , files in os.walk(directory):
        for file in files:
            # Create full path and append to the list
            files_list.append(os.path.join(root, file))

    return files_list


def tokenize(text):
    """pour tokenizer tous les textes présents sur le dictionnaire"""
    max_length = 1000000  # Longueur maximale pour spaCy
    tokens = []
    for i in range(0, len(text), max_length):
        doc = nlp(text[i:i + max_length])
        tokens += [token.text.lower() for token in doc if not token.is_stop]
    return tokens


def preprocess_data(directory):
    """charger les données -> fichier_pdf : text"""
    
    # Fonction pour extraire le texte d'un fichier PDF
    def extract_text_from_pdf(pdf_path):
        try:
            with fitz.open(pdf_path) as doc:
                extracted_text = ''
                for page in doc:
                    extracted_text += page.get_text() + '\n'
        except Exception as e:
            print(f"Erreur lors de l'ouverture du fichier {pdf_path}: {e}")
            return ""

        utf8_text = extracted_text.encode('utf-8')
        return correct_accent(utf8_text.decode('utf-8'))

    all_files = get_all_files(directory)
    index2tokens = {}
    index2file = {}
    index = 0
    for file in all_files:
        print(f"f{file}")
        text = extract_text_from_pdf(file)
        tokens = tokenize(text)
        filename = file.split('/')[-1].split('.')[0]
        index2tokens[index] = tokens
        index2file[index] = file
        index += 1
    return index2tokens , index2file 





def create_inverted_index(directory):
    """ Création de l'index inversé """
    ind2tok , ind2file = preprocess_data(directory)
    inverted_index = defaultdict(list)
    for doc_id, words in ind2tok.items():
        #words = text.lower().split()  # Séparer les mots
        for word in set(words):  # Utiliser un set pour éviter les doublons
            inverted_index[word].append(doc_id)
    return inverted_index , ind2tok , ind2file



def save_data(inverted_index, ind2tok, ind2file, prefix):
    """Enregistrer les données dans des fichiers."""
    with open(f'{prefix}_inverted_index.pkl', 'wb') as f:
        pickle.dump(inverted_index, f)
    with open(f'{prefix}_ind2tok.pkl', 'wb') as f:
        pickle.dump(ind2tok, f)
    with open(f'{prefix}_ind2file.pkl', 'wb') as f:
        pickle.dump(ind2file, f)

def load_data(prefix):
    """Charger les données à partir de fichiers."""
    with open(f'{prefix}_inverted_index.pkl', 'rb') as f:
        inverted_index = pickle.load(f)
    with open(f'{prefix}_ind2tok.pkl', 'rb') as f:
        ind2tok = pickle.load(f)
    with open(f'{prefix}_ind2file.pkl', 'rb') as f:
        ind2file = pickle.load(f)
    
    return inverted_index, ind2tok, ind2file

preprocessed_data_path = str(paths.preprocessed_data)

inverted_index, ind2tok, ind2file = create_inverted_index(paths.data_path)
save_data(inverted_index, ind2tok, ind2file, f"{preprocessed_data_path}/preprocessed_data")

print(f"les données ont été sauvegardés sur le dossier {preprocessed_data_path}")