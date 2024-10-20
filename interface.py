import os
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import global_system
import paths
from textblob import TextBlob  # Importer TextBlob pour la correction orthographique



inverted_index, ind2tok, ind2file , ind2text = global_system.load_data()

def perform_search(correction=False):
    """Utilise la fonction get_documents pour afficher les meilleurs résultats."""
    query = search_entry.get()

    if not query:
        messagebox.showwarning("Input Error", "Please provide a search query.")
        return

    # Correction orthographique avec TextBlob
    corrected_query = str(TextBlob(query).correct())
    
    # Appel à la fonction get_documents pour obtenir les 5 documents les plus pertinents
    if correction:
        results = global_system.get_documents(corrected_query, inverted_index, ind2tok, ind2file , ind2text)
    else:
        results = global_system.get_documents(query, inverted_index, ind2tok, ind2file , ind2text)
    

    # Clear previous results
    result_listbox.delete(0, tk.END)

    if results:
        for result in results:
            result_listbox.insert(tk.END, result.split('/')[1])
    else:
        result_listbox.insert(tk.END, "No documents found.")



def open_file():
    """Ouvre le fichier sélectionné."""
    selected_file = paths.data_path + "/" + result_listbox.get(tk.ACTIVE)
    if selected_file:
        try:
            if os.name == 'nt':  # Windows
                os.startfile(selected_file)
            elif os.name == 'posix':  # macOS ou Linux
                if os.uname().sysname == 'Darwin':  # macOS
                    subprocess.call(['open', selected_file])
                else:  # Linux
                    subprocess.call(['xdg-open', selected_file])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open the file: {e}")


# Création de la fenêtre principale
root = tk.Tk()
root.title("Document Searcher")

# Saisie de la requête de recherche
search_label = tk.Label(root, text="Search Query:")
search_label.grid(row=0, column=0, padx=10, pady=10)
search_entry = tk.Entry(root, width=50)
search_entry.grid(row=0, column=1, padx=10, pady=10)

# Bouton de recherche
search_button = tk.Button(root, text="Search", command=perform_search)
search_button.grid(row=0, column=2, padx=10, pady=10)

# Résultats de la recherche
result_listbox = tk.Listbox(root, width=80, height=20)
result_listbox.grid(row=1, column=0, columnspan=3, padx=10, pady=10)

# Bouton pour ouvrir le fichier sélectionné
open_button = tk.Button(root, text="Open File", command=open_file)
open_button.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

# Lancement de l'interface
if __name__ == '__main__':
    root.mainloop()
