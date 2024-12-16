def get_code_snippet(file_path, start_line, end_line):
    """
    Returns a code snippet from a file based on start and end lines.

    Args:
        file_path (str): The path to the file.
        start_line (int): The starting line number of the code snippet.
        end_line (int): The ending line number of the code snippet.

    Returns:
        list: A list of lines in the code snippet.
    """
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Ensure start_line and end_line are within the bounds of the file
        start_line = max(1, start_line)
        end_line = min(len(lines), end_line)

        # Adjust for zero-based indexing
        start_index = start_line - 1
        end_index = end_line

        snippet = lines[start_index:end_index]

        return snippet
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
