import os
import tkinter as tk
from tkinter import filedialog, messagebox


def search_files(query, directory):
    """Search for files containing the query in the given directory."""
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if query.lower() in file.lower():
                matching_files.append(os.path.join(root, file))
    return matching_files


def browse_directory():
    """Open file dialog to select a directory."""
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        directory_entry.delete(0, tk.END)
        directory_entry.insert(0, folder_selected)


def perform_search():
    """Search for files based on the user's input."""
    query = search_entry.get()
    directory = directory_entry.get()

    if not query or not directory:
        messagebox.showwarning("Input Error", "Please provide both a search query and a directory.")
        return

    results = search_files(query, directory)

    # Clear previous results
    result_listbox.delete(0, tk.END)

    if results:
        for result in results:
            result_listbox.insert(tk.END, result)
    else:
        result_listbox.insert(tk.END, "No files found.")


def open_file():
    """Open the selected file."""
    selected_file = result_listbox.get(tk.ACTIVE)
    if selected_file:
        os.startfile(selected_file)


# Create main window
root = tk.Tk()
root.title("Document Searcher")

# Directory selection
directory_label = tk.Label(root, text="Directory:")
directory_label.grid(row=0, column=0, padx=10, pady=10)
directory_entry = tk.Entry(root, width=50)
directory_entry.grid(row=0, column=1, padx=10, pady=10)
browse_button = tk.Button(root, text="Browse", command=browse_directory)
browse_button.grid(row=0, column=2, padx=10, pady=10)

# Search input
search_label = tk.Label(root, text="Search Query:")
search_label.grid(row=1, column=0, padx=10, pady=10)
search_entry = tk.Entry(root, width=50)
search_entry.grid(row=1, column=1, padx=10, pady=10)

# Search button
search_button = tk.Button(root, text="Search", command=perform_search)
search_button.grid(row=1, column=2, padx=10, pady=10)

# Results listbox
result_listbox = tk.Listbox(root, width=80, height=20)
result_listbox.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

# Open file button
open_button = tk.Button(root, text="Open File", command=open_file)
open_button.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

# Start the GUI loop
if __name__ == '__main__':
    root.mainloop()