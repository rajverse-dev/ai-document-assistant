
from src.retriever import DocumentRetriever
from src.llm_client import OllamaClient


SYSTEM_PROMPT = """
You are an AI document assistant.

Answer the user's question using only the supplied context.

Rules:
1. Do not invent facts or information.
2. If the context does not contain the answer, say:
   "The provided documents do not contain this information."
3. When using information from a context section, cite it
   using its reference, such as [1] or [2].
4. Give clear, concise, human-readable answers.
"""


class RAGPipeline:
    def __init__(
        self,
        retriever: DocumentRetriever,
        llm_client: OllamaClient,
        max_context_chars: int = 6000,
    ):
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be positive")

        self.retriever = retriever
        self.llm_client = llm_client
        self.max_context_chars = max_context_chars

    def _build_context(self, results: list[dict]):
        context_parts = []
        sources = []
        remaining = self.max_context_chars

        for index, result in enumerate(results, start=1):
            metadata = result["metadata"]
            text = result["text"]

            if remaining <= 0:
                break

            text = text[:remaining]
            if not text.strip():
                continue

            source = {
                "reference": index,
                "source": metadata.get("source", "Unknown"),
                "page_number": metadata.get("page_number"),
                "chunk_index": metadata.get("chunk_index"),
                "similarity": result.get("similarity"),
            }

            context_parts.append(
                f"[{index}] Source: {source['source']}\n{text}"
            )
            sources.append(source)
            remaining -= len(text)

        return "\n\n".join(context_parts), sources

    def ask(self, question: str) -> dict:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        # Step 1: Retrieve relevant document chunks
        results = self.retriever.retrieve(question.strip())

        if not results:
            return {
                "question": question,
                "answer": (
                    "I could not find relevant information "
                    "in the uploaded documents."
                ),
                "sources": [],
            }

        # Step 2: Build context from retrieved chunks
        context, sources = self._build_context(results)

        if not context:
            return {
                "question": question,
                "answer": "No usable context was retrieved.",
                "sources": [],
            }

        # Step 3: Build the prompt
        prompt = f"""
Use the following document context to answer the question.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Answer using only the document context.
Include reference numbers such as [1] when citing information.
"""

        # Step 4: Generate an answer with the local LLM
        answer = self.llm_client.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
        )

        # Step 5: Return the answer and source metadata
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }
