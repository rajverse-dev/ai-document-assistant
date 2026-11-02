
from src.llm_client import OllamaClient


def main():

    client = OllamaClient()

    system_prompt = """
    You are an AI document assistant.

    Answer questions using only the information
    supplied by the user.

    If the information is missing, clearly say:
    "The provided context does not contain this information."

    Do not invent facts.
    """

    context = """
    RAG combines information retrieval with language
    generation.

    It retrieves relevant passages from external
    documents and uses them to generate answers.
    """

    question = "What is the capital of Japan?"

    prompt = f"""
    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = client.generate(
        prompt=prompt,
        system_prompt=system_prompt,
    )

    print("Question:", question)
    print("\nAnswer:", response)


if __name__ == "__main__":
    main()
