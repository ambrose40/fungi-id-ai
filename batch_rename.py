# batch_rename.py
import os
import sys
import shutil

def remove_cyrillic_letters(name):
    # List of Cyrillic letters in Unicode
    cyrillic_letters = [chr(i) for i in range(0x0400, 0x04FF + 1)]
    new_name = ""
    for char in name:
        if char not in cyrillic_letters:
            new_name += char
    return new_name

def delete_files_and_folders(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            if not file.endswith(".jpg"):
                os.remove(os.path.join(root, file))
    for root, dirs, files in os.walk(path, topdown=False):
        if not dirs and not files:
            os.rmdir(root)


def rename_folders(dirpath):
    for dirname in os.listdir(dirpath):
        old_dir = os.path.join(dirpath, dirname)
        
        if os.path.isdir(old_dir):
            if dirname == ".jalbum":
                print(f"'{old_dir}'")
                shutil.rmtree(old_dir)
            try:
                new_name = dirname.replace(" - ", "").replace(" , ", "").replace(" ,", "").replace(", ", "").replace(",", "").replace(" -", "").replace("- ", "").replace("-", "").replace("  ", "").lower().lstrip()
                new_dir = os.path.join(dirpath, remove_cyrillic_letters(new_name))
                if (new_dir != old_dir):
                    os.rename(old_dir, new_dir)
                delete_files_and_folders(dirpath)
                if os.path.isdir(new_dir):
                    rename_folders(new_dir)
            except Exception as e:
                print(f"Error while renaming '{dirname}': {e}")

rename_folders("D:\Fungarium.backup")