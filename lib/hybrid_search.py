import os

from .keyword_search import InvertedIndex, load_movies
from .semantic_search import ChunkedSemanticSearch
from config import settings
from pydantic import BaseModel


class DocumentStat(BaseModel):
    document: dict
    bm25_score: float
    semantic_score: float


class DocumentRank(BaseModel):
    document: dict
    bm25_rank: int | None
    semantic_rank: int | None


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    if type(semantic_score) == dict:
        print(semantic_score)
        raise ("Ошибка")
    return alpha * bm25_score + (1 - alpha) * semantic_score


def weighted_search_command(query: str, alpha: float, limit: int):
    documents = load_movies(settings.MOVIE_ADDRESS)
    hybrid_search = HybridSearch(documents)
    result = hybrid_search.weighted_search(query, alpha, limit)
    for index, (doc_id, hybrid_score) in enumerate(result, start=1):
        print(
            f"{index}. {[doc_id]} {hybrid_search.idx.docmap[doc_id]["title"]} ({hybrid_score:.4f})"
        )

def rrf_search_command(query: str, k: int, limit: int):
    documents = load_movies(settings.MOVIE_ADDRESS)
    hybrid_search = HybridSearch(documents)
    result = hybrid_search.rrf_search(query, k, limit)
    for index, (doc_id, hybrid_score) in enumerate(result, start=1):
        print(
            f"{index}. {[doc_id]} {hybrid_search.idx.docmap[doc_id]["title"]} ({hybrid_score:.4f})"
        )

def normalize_command(scores: list[tuple[int, float]]):
    if not scores:
        return
    max_score = max(scores, key=lambda itm: itm[1])
    min_score = min(scores, key=lambda itm: itm[1])
    n = len(scores)

    if max_score == min_score:
        return [1.0] * n

    normalize_scores = [0] * n

    for i in range(n):
        doc_id = scores[i][0]
        normalize_scores[i] = (
            doc_id,
            (scores[i] - min_score) / (max_score - min_score),
        )

    return normalize_scores

def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex(
            settings.STOPWORDS_ADDRESS,
            settings.MOVIE_ADDRESS,
            settings.INDEX_CACHE,
            settings.DOCMAP_CACHE,
            settings.TF_CACHE,
            settings.DOCLENGTHS_CACHE,
        )

        if not os.path.exists(self.idx.index_cache_f):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_scores = self._bm25_search(query, 500)
        semantic_scores = self.semantic_search.search_chunks(query, 500)

        normalized_bm25 = self.normalize_scores(bm25_scores)
        docmap: dict[int, DocumentStat] = (
            dict()
        )  # doc_id -> {"document":..., "bm25_score":..., "semantic_score":...}
        doc_data: DocumentStat = (
            dict()
        )  # {"document":..., "bm25_score":..., "semantic_score":...}

        for document_id, document in self.idx.docmap.items():
            doc_data = DocumentStat(
                document=document, bm25_score=0.0, semantic_score=0.0
            )
            docmap[document_id] = doc_data

        for doc_id, score in normalized_bm25:
            doc_data = docmap[doc_id]
            doc_data.bm25_score = score

        for semantic_score in semantic_scores:
            doc_id = semantic_score["id"]
            doc_data = docmap[doc_id]
            doc_data.semantic_score = semantic_score["score"]

        doc_hybrid_score: dict[int, float] = dict()  # doc_id -> hybrid_score

        for doc_id, doc_data in docmap.items():
            bm25_score = doc_data.bm25_score
            semantic_score = doc_data.semantic_score
            doc_hybrid_score[doc_id] = hybrid_score(bm25_score, semantic_score, alpha)

        result = sorted(
            doc_hybrid_score.items(), key=lambda item: item[1], reverse=True
        )

        return result[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

    def normalize_scores(self, scores: list[tuple[int, float]]):
        if not scores:
            return
        max_score = max(scores, key=lambda itm: itm[1])[1]
        min_score = min(scores, key=lambda itm: itm[1])[1]
        n = len(scores)

        if max_score == min_score:
            return [1.0] * n

        normalize_scores: list[tuple] = list()

        for i in range(n):
            doc_id = scores[i][0]

            normalize_scores.append(
                (doc_id, (scores[i][1] - min_score) / (max_score - min_score))
            )

        return normalize_scores

    def rrf_search(self, query: str, k: int, limit: int):
        bm25_results = self._bm25_search(query, limit=500)
        bm25_ranks = [doc_id for doc_id, _ in bm25_results]

        chunked_search_results = self.semantic_search.search_chunks(query, limit=500)
        chunked_search_ranks = [result["id"] for result in chunked_search_results]

        doc_ranks: dict[int, DocumentRank] = dict()

        for doc_id in set(bm25_ranks + chunked_search_ranks):
            doc_ranks[doc_id] = DocumentRank(document=self.idx.docmap[doc_id], bm25_rank=None, semantic_rank=0)

        for rank, doc_id in enumerate(bm25_ranks, start=1):
            doc_rank = doc_ranks[doc_id]
            doc_rank.bm25_rank = rank
            
        for rank, doc_id in enumerate(chunked_search_ranks, start=1):
            doc_rank = doc_ranks[doc_id]
            doc_rank.semantic_rank = rank


        documents_rrf: dict[int, float] = dict()
        
        for doc_id, doc_rank in doc_ranks.items():
            if doc_rank.semantic_rank and doc_rank.bm25_rank:
                doc_rrf = rrf_score(doc_rank.bm25_rank, k) + rrf_score(doc_rank.semantic_rank, k)
                documents_rrf[doc_id] = doc_rrf

        result = sorted([item for item in documents_rrf.items()], key=lambda item: item[1], reverse=True)
        return result[:limit]