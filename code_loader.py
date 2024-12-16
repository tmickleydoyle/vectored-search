import os
from tree_sitter import Parser

from config import INCLUDED_EXTENSIONS, LANGUAGE_PARSERS, PYTHON_LANGUAGE


def load_code_files(directory, included_extensions=INCLUDED_EXTENSIONS):
    """
    Recursively loads code files with specified extensions from a directory and parses them into meaningful components.

    Args:
        directory (str): Path to the root directory.
        included_extensions (tuple): File extensions to include (e.g., (".py",)).

    Returns:
        List[Dict[str, Any]]: List of extracted code sections with type, content, row number, etc.
    """
    code_snippets = []

    for root, _, files in os.walk(directory):
        for filename in files:
            # Exclude hidden files and filter by included extensions
            if not filename.startswith(".") and filename.endswith(included_extensions):
                file_path = os.path.join(root, filename)
                extension = os.path.splitext(filename)[1]
                parser_language = LANGUAGE_PARSERS.get(extension)

                if parser_language:
                    try:
                        with open(file_path, "r", encoding="utf-8") as file:
                            code = file.read()

                        # Parse the file with Tree-sitter
                        parser = Parser(PYTHON_LANGUAGE)
                        tree = parser.parse(code.encode())

                        # Extract meaningful components
                        snippets = extract_code_sections(
                            tree, code, filename, file_path
                        )
                        code_snippets.extend(snippets)
                    except (OSError, UnicodeDecodeError) as e:
                        print(f"Error reading file {file_path}: {e}")

    return code_snippets


def extract_code_sections(tree, code, filename, file_path):
    """
    Extracts functions, classes, and other meaningful sections from the parsed AST.

    Args:
        tree (tree_sitter.Node): Parsed AST tree.
        code (str): Original code content.
        filename (str): Name of the file.
        file_path (str): Full file path.

    Returns:
        List[Dict[str, Any]]: List of extracted sections with metadata.
    """
    root_node = tree.root_node
    snippets = []

    def traverse(node):
        if node.type in [
            "function_definition",
            "class_definition",
        ]:  # Adjust types as needed
            start_row = node.start_point[0] + 1  # Tree-sitter uses 0-based indexing
            end_row = node.end_point[0] + 1
            snippet = "\n".join(code.splitlines()[start_row - 1 : end_row])
            snippets.append(
                {
                    "type": node.type,
                    "content": snippet,
                    "start_row": start_row,
                    "end_row": end_row,
                    "filename": filename,
                    "file_path": file_path,
                }
            )

        # Traverse child nodes
        for child in node.children:
            traverse(child)

    traverse(root_node)
    return snippets
