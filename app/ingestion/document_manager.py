def list_documents(vector_store):
    """Return unique indexed documents and their chunk counts."""

    collection = vector_store.collection

    result = collection.get(
        include=["metadatas"]
    )

    documents = {}

    for metadata in result.get("metadatas") or []:
        if not metadata:
            continue

        document_id = metadata.get("document_id")
        filename = metadata.get("filename")

        if not document_id or not filename:
            continue

        if document_id not in documents:
            documents[document_id] = {
                "document_id": document_id,
                "filename": filename,
                "file_type": metadata.get("file_type"),
                "chunks_count": 0,
            }

        documents[document_id]["chunks_count"] += 1

    return list(documents.values())