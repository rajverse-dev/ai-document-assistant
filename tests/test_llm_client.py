
from src.llm_client import OllamaClient


def main():

    client = OllamaClient(
        model="qwen2.5:3b",
    )

    prompt = """
    Explain Retrieval-Augmented Generation (RAG)
    in simple English.

    Give:
    1. Definition
    2. How it works
    3. One real-world example
    """

    print("Sending request to Ollama...\n")

    response = client.generate(prompt)

    print("Generated Answer")
    print("=" * 50)

    print(response)

    assert response.strip()

    print("\nLLM integration test passed!")


if __name__ == "__main__":
    main()
