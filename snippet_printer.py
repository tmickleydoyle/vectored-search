def pprint_code_snippet(file_path, row_number):
    """
    Pretty-prints a code snippet from a file.
    """
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()

        start = max(0, row_number - 4)
        end = min(len(lines), row_number + 6)
        snippet = lines[start:end]

        if start < row_number - 1 and snippet[0].strip() == "":
            snippet.pop(0)

        print(f"File: {file_path}")
        for line in snippet:
            print(line, end="")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
