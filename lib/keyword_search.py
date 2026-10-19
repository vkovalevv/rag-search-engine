import json
import os
import string
import math

from pickle import dump, load
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
from config import settings
from decimal import Decimal

BM25_K1 = settings.BM25_K1
BM25_B = settings.BM25_B

stemmer = PorterStemmer()


def load_stop_words(file_path: str):
    with open(os.path.abspath(file_path)) as f:
        stop_words = set(f.read().splitlines())
        return {
            word.lower().translate({ord(c): None for c in string.punctuation})
            for word in stop_words
        }


def remove_stop_words(query_tokens: list[str], stop_words: list[str]):
    result_tokens = []
    for token in query_tokens:
        if token in stop_words:
            continue
        result_tokens.append(token)
    return result_tokens


def tokenize_text(text: str, stop_words: list[str]) -> list[str]:
    text = text.lower()  # case sensitivity
    without_punctuation = text.translate(
        {ord(c): None for c in string.punctuation}
    )  # punctuation
    tokens_without_punctuation = without_punctuation.split()  # tokenization
    tokens_without_stopw = remove_stop_words(tokens_without_punctuation, stop_words)
    tokens = [stemmer.stem(token) for token in tokens_without_stopw]
    return tokens


def tokenize_term(term: str, stop_words: list[str]) -> str:
    token_term = tokenize_text(term, stop_words)
    if len(token_term) != 1:
        raise ValueError
    return token_term[0]


def tokens_match(title_tokens: list[str], query_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def load_movies(file_path: str):
    with open(os.path.abspath(file_path)) as f:
        movies = json.load(f)["movies"]
    return movies


class InvertedIndex:
    def __init__(
        self,
        stop_words_file: str,
        data_file: str,
        index_cache_f: str,
        docmap_cache_f: str,
        tf_cache_f: str,
        doclengths_cache_f: str,
    ) -> None:
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths: dict[int, int] = {}

        self.stop_words = load_stop_words(stop_words_file)

        self.data_file = data_file
        self.index_cache_f = index_cache_f
        self.docmap_cache_f = docmap_cache_f
        self.tf_cache_f = tf_cache_f
        self.doclengths_cache_f = doclengths_cache_f

    def __add_document(self, doc_id, text) -> None:
        tokens = tokenize_text(text, self.stop_words)
        document_tf = Counter()

        for token in tokens:
            self.index[token].add(doc_id)
            document_tf[token] += 1

        self.term_frequencies[doc_id] = document_tf
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(value for value in self.doc_lengths.values()) / len(self.doc_lengths)

    def get_documents(self, term):
        return sorted(self.index[term])

    def build(self):
        movies = load_movies(self.data_file)
        for movie in movies:
            movie_id = movie["id"]
            movie_text = movie["title"] + " " + movie["description"]
            self.__add_document(movie_id, movie_text)
            self.docmap[movie_id] = movie

    def load(self):
        try:
            with open(os.path.abspath(self.index_cache_f), mode="rb") as index_file:
                self.index = load(index_file)

            with open(os.path.abspath(self.docmap_cache_f), mode="rb") as docmap_file:
                self.docmap = load(docmap_file)

            with open(os.path.abspath(self.tf_cache_f), mode="rb") as tf_file:
                self.term_frequencies = load(tf_file)

            with open(
                os.path.abspath(self.doclengths_cache_f), mode="rb"
            ) as doc_lengths_file:
                self.doc_lengths = load(doc_lengths_file)

        except FileNotFoundError as e:
            raise

    def save(self):
        with open(os.path.abspath(self.index_cache_f), mode="wb") as index_file:
            dump(self.index, index_file)

        with open(os.path.abspath(self.docmap_cache_f), mode="wb") as docmap_file:
            dump(self.docmap, docmap_file)

        with open(os.path.abspath(self.tf_cache_f), mode="wb") as tf_file:
            dump(self.term_frequencies, tf_file)

        with open(
            os.path.abspath(self.doclengths_cache_f), mode="wb"
        ) as doc_lengths_file:
            dump(self.doc_lengths, doc_lengths_file)

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies[doc_id].get(term, 0)

    def get_idf(self, term: str):
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(term))
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = len(self.get_documents(term))
        bm_idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return bm_idf

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / avg_doc_length)
        bm25_tf = tf * (k1 + 1) / (tf + k1 * length_norm)
        return bm25_tf

    def bm25(self, doc_id: int, term: str) -> Decimal:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int):
        tokens = tokenize_text(query, self.stop_words)
        doc_scores = defaultdict(float)
        for token in tokens:
            docs = self.get_documents(token)
            for doc_id in docs:
                bm25_score = self.bm25(doc_id, token)
                doc_scores[doc_id] += bm25_score

        doc_scores = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        return doc_scores[:limit]
