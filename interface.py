import os
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import global_system
import paths
from textblob import TextBlob  # Import TextBlob for spell correction

# Load the inverted index and other data from the system
inverted_index, ind2tok, ind2file , ind2text = global_system.load_data()

def perform_search(correction=False):
    """Uses the get_documents function to display the best results."""
    query = search_entry.get()

    if not query:
        messagebox.showwarning("Input Error", "Please provide a search query.")
        return

    # Spell correction using TextBlob
    corrected_query = str(TextBlob(query).correct())
    
    # Call the get_documents function to get the top 5 relevant documents
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
    """Opens the selected file."""
    selected_file = paths.data_path + "/" + result_listbox.get(tk.ACTIVE)
    if selected_file:
        try:
            if os.name == 'nt':  # Windows
                os.startfile(selected_file)
            elif os.name == 'posix':  # macOS or Linux
                if os.uname().sysname == 'Darwin':  # macOS
                    subprocess.call(['open', selected_file])
                else:  # Linux
                    subprocess.call(['xdg-open', selected_file])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open the file: {e}")


# Create the main window
root = tk.Tk()
root.title("Document Searcher")

# Search query input
search_label = tk.Label(root, text="Search Query:")
search_label.grid(row=0, column=0, padx=10, pady=10)
search_entry = tk.Entry(root, width=50)
search_entry.grid(row=0, column=1, padx=10, pady=10)

# Search button
search_button = tk.Button(root, text="Search", command=perform_search)
search_button.grid(row=0, column=2, padx=10, pady=10)

# Search results
result_listbox = tk.Listbox(root, width=80, height=20)
result_listbox.grid(row=1, column=0, columnspan=3, padx=10, pady=10)

# Button to open the selected file
open_button = tk.Button(root, text="Open File", command=open_file)
open_button.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

# Launch the interface
if __name__ == '__main__':
    root.mainloop()
