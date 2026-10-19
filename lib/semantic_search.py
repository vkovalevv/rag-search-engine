import re
import os
import json
import string
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
    for index, result in enumerate(search_result, start=1):
        print(f"{index}. {result["title"]} ({result["score"]})")
        print(f"{result["description"]}")


def embed_chunks_command():
    with open(settings.MOVIE_ADDRESS) as f:
        documents = json.load(f)["movies"]
    semantic_chunking = ChunkedSemanticSearch()
    embeddings = semantic_chunking.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")


def search_chunked_command(query: str, limit: int):
    semantic_chunking_search = ChunkedSemanticSearch()
    with open(settings.MOVIE_ADDRESS) as f:
        documents = json.load(f)["movies"]
    embeddings = semantic_chunking_search.load_or_create_chunk_embeddings(documents)

    search_result = semantic_chunking_search.search_chunks(query, limit)

    for i, doc in enumerate(search_result, start=1):
        print(f"\n{i}. {doc["title"]} (score: {doc["score"]:.4f})")
        print(f"   {doc["description"]}...")


class SemanticSearch:
    def __init__(self, model_name: str = settings.TRANSFORMER_MODEL) -> None:
        self.model: SentenceTransformer = SentenceTransformer(model_name)
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

        similarity_result.sort(key=tuple_sort, reverse=True)

        similarity_result = similarity_result[:limit]
        return [
            {
                "score": score,
                "title": document["title"],
                "description": document["description"],
            }
            for (score, document) in similarity_result
        ]


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


def chunking(text: str, size: int, overlap: int):
    words = text.split()
    first_ptr, second_ptr = 0, size
    chunks = []

    while second_ptr < len(words):
        chunk = words[first_ptr:second_ptr]

        chunks.append(" ".join(chunk))
        first_ptr += size - overlap
        second_ptr += size - overlap

    remainder = " ".join(words[first_ptr:])

    chunks.append(remainder)

    return chunks


def semantic_chunk(text: str, max_chunk_size: int, overlap: int):
    text = text.strip()
    if not text:
        return []

    split_result = re.split(r"(?<=[.!?])\s+", text)
    
    if len(split_result) == 1:
        sentence = split_result[0]
        is_ends_punct = sentence.endswith(tuple(c for c in string.punctuation))
        if not is_ends_punct:
            sentence = sentence + "."
            return [sentence]
        
    first_ptr, second_ptr = 0, max_chunk_size
    chunks = []

    while second_ptr < len(split_result):
        sentences = split_result[first_ptr:second_ptr]

        chunks.append(" ".join(sentence.strip() for sentence in sentences if sentence.strip()))

        first_ptr += max_chunk_size - overlap
        second_ptr += max_chunk_size - overlap

    remainder = " ".join(split_result[first_ptr:])

    chunks.append(remainder)

    return chunks


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = list()
        self.chunk_metadata = dict() 
        self.all_chunks: list[str] = []
        self.idx_chunk: dict[int, str] = dict()

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        for document in documents:
            doc_id = document["id"]
            self.document_map[doc_id] = document

        all_chunks: list[str] = []
        chunk_index = 0

        for document in documents:
            if not document["description"]:
                continue
            description = document["description"]
            movie_idx = document["id"]
            descrption_chunks = semantic_chunk(
                description, max_chunk_size=4, overlap=1
            )
            all_chunks.extend(descrption_chunks)

            for index, chunk in enumerate(descrption_chunks):
                self.chunk_metadata[chunk_index] = {
                    "movie_idx": movie_idx,
                    "chunk_idx": index,
                    "total_chunks": len(descrption_chunks),
                }
                chunk_index += 1

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)

        with open(settings.EMBEDDINGS_CACHE, "wb") as f:
            np.save(f, self.chunk_embeddings)

        with open(settings.CHUNK_METADATA_CACHE, "w") as f:
            json.dump(
                {"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)},
                f,
                indent=2,
            )

        return all_chunks

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        for document in documents:
            doc_id = document["id"]
            self.document_map[doc_id] = document

        embedding_path = os.path.abspath(settings.EMBEDDINGS_CACHE)
        chunk_metadata_path = os.path.abspath(settings.CHUNK_METADATA_CACHE)
        is_embedding_exists = os.path.exists(embedding_path)
        is_metadata_exists = os.path.exists(chunk_metadata_path)

        if is_embedding_exists and is_metadata_exists:
            with open(embedding_path, "rb") as embedding_file:
                self.chunk_embeddings = np.load(embedding_file)
            with open(chunk_metadata_path, "r") as metadata_file:
                self.chunk_metadata = json.load(metadata_file)["chunks"]
            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        query_embedding = self.generate_embedding(query)
        chunks_scores = []
        for index, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity_score = cosine_similarity(query_embedding, chunk_embedding)
            metadata = self.chunk_metadata[str(index)]
            
            chunk_score = {
                "movie_idx": metadata["movie_idx"],
                "chunk_idx": metadata["chunk_idx"],
                "score": similarity_score,
            }
            chunks_scores.append(chunk_score)

        movie_score = dict()

        for chunk_score in chunks_scores:
            movie_idx = chunk_score["movie_idx"]
            movie_score[movie_idx] = max(
                chunk_score["score"], movie_score.get(movie_idx, 0)
            )

        def order_result(movie_result: tuple):
            movie_idx, movie_score = movie_result
            return movie_score

        id_score_result = sorted(
            [
                (movie_idx, chunk_score)
                for movie_idx, chunk_score in movie_score.items()
            ],
            key=order_result,
            reverse=True,
        )[:limit]

        result = []

        for movie_idx, score in id_score_result:
            doc = self.document_map[movie_idx]
            result.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "description": doc["description"][:100],
                    "score": round(score, 3),
                }
            )

        return result
