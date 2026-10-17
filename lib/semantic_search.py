import re
import os
import json
from sentence_transformers import SentenceTransformer
from config import settings
import numpy as np


def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def verify_embeddings():
    semantic_search = SemanticSearch()
    with open(settings.MOVIE_ADDRESS, "rb") as f:
        documents = json.load(f)["movies"]
    embeddings = semantic_search.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    return embedding

def embed_query_text(query: str):
    query = query.strip()
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    return embedding

def search_command(query: str, limit: int):
    semantic_search = SemanticSearch()

    with open(settings.MOVIE_ADDRESS, "rb") as f:
        documents = json.load(f)["movies"]

    semantic_search.load_or_create_embeddings(documents)

    search_result = semantic_search.search(query, limit)
    for (index,result) in enumerate(search_result, start=1):
        print(f"{index}. {result["title"]} ({result["score"]})")
        print(f"{result["description"]}")

class SemanticSearch:
    def __init__(self) -> None:
        self.model: SentenceTransformer = SentenceTransformer(
            settings.TRANSFORMER_MODEL
        )
        self.embeddings = None
        self.documents = None
        self.document_map = dict()

    def generate_embedding(self, text):
        if not text.split():
            raise ValueError
        embedding = self.model.encode([text])[0]
        return embedding

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        representations = []

        for document in documents:
            doc_id = document["id"]
            self.document_map[doc_id] = document
            representations.append(f"{document['title']}: {document['description']}")

        self.embeddings = self.model.encode(representations, show_progress_bar=True)

        np.save(settings.EMBEDDINGS_CACHE, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        representations = []

        for document in documents:
            doc_id = document["id"]
            self.document_map[doc_id] = document
            representations.append(f"{document['title']}: {document['description']}")

        file_path = os.path.abspath(settings.EMBEDDINGS_CACHE)
        is_exist = os.path.exists(file_path)

        if is_exist:
            self.embeddings = np.load(file_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        else:
            return self.build_embeddings(documents)

    def search(self, query: str, limit: int) -> list[dict]:
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embed = embed_query_text(query)
        similarity_result = []

        for doc_id, doc_embed in enumerate(self.embeddings, start=1):
            similarity = cosine_similarity(query_embed, doc_embed)
            document = self.document_map[doc_id]
            similarity_result.append((similarity, document))

        similarity_result.sort(key=tuple_sort,reverse=True)

        similarity_result = similarity_result[:limit]
        return [{
            "score": score,
            "title": document["title"],
            "description": document["description"]
        } for (score, document) in similarity_result]

def tuple_sort(similarity: tuple):
    score, document = similarity
    return score

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def chunking(text: str, size: int, overlap:int):
    words = text.split()
    first_ptr, second_ptr = 0, size
    chunks = []

    while second_ptr< len(words):
        chunk = words[first_ptr:second_ptr]

        chunks.append(" ".join(chunk))
        first_ptr += size - overlap
        second_ptr += size - overlap

    remainder = " ".join(words[first_ptr:])
    
    chunks.append(remainder)

    print(f"Chunking {len(text)} characters")
    for index, chunk in enumerate(chunks, start=1):
        print(f"{index}. {chunk}")

    return chunks

def semantic_chunking(text:str, max_chunk_size: int, overlap:int):
    split_result = re.split(r"(?<=[.!?])\s+", text)
    first_ptr, second_ptr = 0, max_chunk_size
    chunks = []
    previous_chunk = None
    chunk_overlap = ""

    while second_ptr < len(split_result):
        chunk = None
  
        chunk = split_result[first_ptr:second_ptr]

        chunks.append(" ".join(chunk))

        first_ptr += max_chunk_size - overlap
        second_ptr += max_chunk_size - overlap

    remainder = " ".join(split_result[first_ptr:])

    chunks.append(remainder)

    print(f"Semantically chunking {len(text)} characters")
    for index, chunk in enumerate(chunks, start=1):
        print(f"{index}. {chunk}")

    return chunks