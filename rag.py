import ollama


def rag(query, context):
    """
    Generate a response for the given query and context.

    Args:
        query (str): The question to ask.
        context (str): The context to provide.
        verbose (bool): Whether to print the generated response.

    Returns:
        str: The generated response
    """
    response_question = ollama.generate(
        model="qwen2.5-coder:1.5b",
        prompt=f"""
        Given the context:
            {context}

        Answer the question only using the context:
            {query}
        """,
    )
    return response_question
