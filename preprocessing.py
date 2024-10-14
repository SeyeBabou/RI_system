import os
import fitz
import spacy


#corriger les accents
def correct_accent(text):
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
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Create full path and append to the list
            files_list.append(os.path.join(root, file))

    return files_list

#charger les données
def get_data(all_files):

  # Fonction pour extraire le texte d'un fichier PDF
  def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        extracted_text = ''
        for page in doc:
            extracted_text += page.get_text() + '\n'

    # Ensure the text is in UTF-8 format
    utf8_text = extracted_text.encode('utf-8')
    return correct_accent(utf8_text.decode('utf-8'))

  data = {}
  for file in all_files:
    text = extract_text_from_pdf(file)
    filename = file.split('/')[-1].split('.')[0]
    data[filename] = text
  return data

#tokenization
def tokenize(directory):
    no_tokenize_data = get_data(get_all_files(directory))
    for filename in no_tokenize_data.keys():
        text = no_tokenize_data[filename]

