
from io import BytesIO

import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model():
    """Load the model used for semantic search."""
    return SentenceTransformer(EMBEDDING_MODEL)


def extract_text(uploaded_files):
    """Extract text from uploaded PDF and TXT files."""
    documents = []

    for file in uploaded_files:
        filename = file.name
        content = file.getvalue()

        if filename.lower().endswith(".pdf"):
            reader = PdfReader(BytesIO(content))
            text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        elif filename.lower().endswith(".txt"):
            text = content.decode("utf-8", errors="replace")
        else:
            continue

        if text.strip():
            documents.append({
                "source": filename,
                "text": text
            })

    return documents


def chunk_documents(documents, chunk_size=800, overlap=120):
    """Split documents into overlapping character chunks."""
    chunks = []

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    for document in documents:
        text = document["text"]
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            passage = text[start:end].strip()

            if passage:
                chunks.append({
                    "source": document["source"],
                    "text": passage
                })

            if end == len(text):
                break

            start = end - overlap

    return chunks


def build_index(chunks, model):
    """Build BM25 and vector-search indexes."""
    if not chunks:
        raise ValueError("No text chunks found. Upload readable documents.")

    tokenized_chunks = [
        chunk["text"].lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return bm25, np.asarray(embeddings)


def retrieve(query, chunks, model, bm25, embeddings, top_k=5):
    """Combine keyword and semantic retrieval using RRF."""
    if not query.strip():
        return []

    if not chunks:
        return []

    top_k = max(1, min(top_k, len(chunks)))

    # 1. Keyword search with BM25
    tokenized_query = query.lower().split()
    keyword_scores = bm25.get_scores(tokenized_query)

    keyword_order = np.argsort(keyword_scores)[::-1][:top_k]

    # 2. Semantic vector search
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    vector_scores = embeddings @ query_embedding
    vector_order = np.argsort(vector_scores)[::-1][:top_k]

    # 3. Reciprocal Rank Fusion
    rrf_scores = {}

    for rank, idx in enumerate(keyword_order, start=1):
        rrf_scores[int(idx)] = (
            rrf_scores.get(int(idx), 0) + 1 / (60 + rank)
        )

    for rank, idx in enumerate(vector_order, start=1):
        rrf_scores[int(idx)] = (
            rrf_scores.get(int(idx), 0) + 1 / (60 + rank)
        )

    ranked_indices = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )[:top_k]

    results = []

    for idx in ranked_indices:
        results.append({
            "source": chunks[idx]["source"],
            "text": chunks[idx]["text"],
            "rrf_score": round(rrf_scores[idx], 5),
            "bm25_score": round(float(keyword_scores[idx]), 4),
            "vector_score": round(float(vector_scores[idx]), 4)
        })

    return results
