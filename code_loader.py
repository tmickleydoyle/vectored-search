import os

def load_code_files(directory):
    """
    Recursively loads Python files from a directory.

    Args:
        directory (str): Path to the root directory.

    Returns:
        List[Tuple[str, str, int, str]]: List of filename, snippet, row number, and file path.
    """
    python_files = []

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".py"):
                file_path = os.path.join(root, filename)
                with open(file_path, "r") as file:
                    lines = file.readlines()
                    for i in range(len(lines) - 4):
                        snippet = "".join(lines[i:i + 5])
                        row_number = i + 1
                        python_files.append((filename, snippet, row_number, file_path))

    return python_files
