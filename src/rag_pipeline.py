
from src.retriever import DocumentRetriever
from src.llm_client import OllamaClient


SYSTEM_PROMPT = """
You are an AI document assistant.

Answer the user's question using only the supplied context.

Rules:
1. Do not invent facts or information.
2. If the context does not contain the answer, say:
   "The provided documents do not contain this information."
3. Cite information using the supplied reference numbers,
   such as [1] or [2].
4. Do not create fake references.
5. Give clear, concise, human-readable answers.
6. If the context contains conflicting information,
   explain the conflict instead of choosing arbitrarily.
"""


class RAGPipeline:
    """
    Complete Retrieval-Augmented Generation pipeline.

    Workflow:
    1. Retrieve relevant document chunks.
    2. Build context from retrieved chunks.
    3. Generate an answer using the LLM.
    4. Return answer and source metadata.
    """

    def __init__(
        self,
        retriever: DocumentRetriever,
        llm_client: OllamaClient,
        max_context_chars: int = 6000,
    ):

        if max_context_chars <= 0:
            raise ValueError(
                "max_context_chars must be positive"
            )

        self.retriever = retriever
        self.llm_client = llm_client
        self.max_context_chars = max_context_chars

    def _build_context(
        self,
        results: list[dict],
    ) -> tuple[str, list[dict]]:

        context_parts = []
        sources = []

        remaining = self.max_context_chars

        for index, result in enumerate(results, start=1):

            if remaining <= 0:
                break

            metadata = result["metadata"]
            text = result["text"]

            text = text[:remaining].strip()

            if not text:
                continue

            source = {
                "reference": index,
                "source": metadata.get(
                    "source",
                    "Unknown"
                ),
                "filename": metadata.get(
                    "filename",
                    metadata.get("source", "Unknown")
                ),
                "document_id": metadata.get(
                    "document_id"
                ),
                "page_number": metadata.get(
                    "page_number"
                ),
                "chunk_index": metadata.get(
                    "chunk_index"
                ),
                "similarity": result.get(
                    "similarity"
                ),
            }

            context_parts.append(
                f"""
[{index}]
Source: {source['source']}
Content:
{text}
"""
            )

            sources.append(source)

            remaining -= len(text)

        return "\n\n".join(context_parts), sources

    def ask(
        self,
        question: str,
        top_k: int | None = None,
    ) -> dict:

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty"
            )

        question = question.strip()

        # Step 1: Retrieve relevant document chunks
        results = self.retriever.retrieve(
            question,
            top_k=top_k,
        )

        if not results:
            return {
                "question": question,
                "answer": (
                    "I could not find relevant information "
                    "in the uploaded documents."
                ),
                "sources": [],
                "retrieved_chunks": 0,
            }

        # Step 2: Build context
        context, sources = self._build_context(
            results
        )

        if not context:
            return {
                "question": question,
                "answer": "No usable context was retrieved.",
                "sources": [],
                "retrieved_chunks": 0,
            }

        # Step 3: Build LLM prompt
        prompt = f"""
Use the following document context to answer
the user's question.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Answer using only the supplied document context.
Include reference numbers such as [1] when citing information.
"""

        # Step 4: Generate answer
        answer = self.llm_client.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
        )

        # Step 5: Return result
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(sources),
        }