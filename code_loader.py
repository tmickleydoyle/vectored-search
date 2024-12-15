import os

from config import INCLUDED_EXTENSIONS

def load_code_files(directory, included_extensions=INCLUDED_EXTENSIONS):
    """
    Recursively loads code files with specified extensions from a directory, excluding hidden files.

    Args:
        directory (str): Path to the root directory.
        included_extensions (tuple): File extensions to include (e.g., (".py",)).

    Returns:
        List[Tuple[str, str, int, str]]: List of filename, snippet, row number, and file path.
    """
    code_files = []

    for root, _, files in os.walk(directory):
        for filename in files:
            # Exclude hidden files and filter by included extensions
            if not filename.startswith('.') and filename.endswith(included_extensions):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as file:
                        lines = file.readlines()
                        # Extract 5-line snippets
                        for i in range(len(lines) - 4):
                            snippet = "".join(lines[i:i + 5])
                            row_number = i + 1
                            code_files.append((filename, snippet, row_number, file_path))
                except (OSError, UnicodeDecodeError) as e:
                    print(f"Error reading file {file_path}: {e}")

    return code_files
