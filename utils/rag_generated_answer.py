import ollama
from utils.search.cosine_search_process_1 import cosine_search_1


def create_answer(query, model="llama3"):

    # Retrieve relevant chunks
    df = cosine_search_1(query)

    retrieved_chunks = df[["title", "number", "text"]].to_dict("records")

    # Build context
    context = ""

    for chunk in retrieved_chunks:
        context += f"""
Video: {chunk['title']}
Chunk: {chunk['number']}
Text: {chunk['text']}

"""

    # Prompt for LLM
    prompt = f"""
You are a helpful assistant.

Use the provided context to answer the question.

If the answer is not present in the context, say "I don't know".

Question:
{query}

Context:
{context}

Answer:
"""

    response = ollama.generate(
        model=model,
        prompt=prompt
    )

    return response["response"]