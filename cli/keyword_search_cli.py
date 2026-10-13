import argparse
import json
import os
import string
import math

from pickle import dump, load
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer

MOVIE_ADDRESS = "data/movies.json"
STOPWORDS_ADDRESS = "data/stopwords.txt"
INDEX_CACHE = "cache/index.pkl"
DOCMAP_CACHE = "cache/docmap.pkl"
TF_CACHE = "cache/tf.pkl"


stemmer = PorterStemmer()


def load_stop_words(file_path: str):
    with open(os.path.abspath(file_path)) as f:
        stop_words = set(f.read().splitlines())
    return stop_words

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
    def __init__(self, stop_words: set[str]) -> None:
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.stop_words = stop_words

    def __add_document(self, doc_id, text) -> None:
        tokens = tokenize_text(text, self.stop_words)
        document_tf = Counter()

        for token in tokens:
            self.index[token].add(doc_id)
            document_tf[token] += 1

        self.term_frequencies[doc_id] = document_tf

    def get_documents(self, term):
        return sorted(self.index[term])

    def build(self):
        movies = load_movies(MOVIE_ADDRESS)
        for movie in movies:
            movie_id = movie["id"]
            movie_text = movie["title"] + " " + movie["description"]
            self.__add_document(movie_id, movie_text)
            self.docmap[movie_id] = movie

    def load(self):
        try:
            with open(os.path.abspath(INDEX_CACHE), mode="rb") as index_file:
                self.index = load(index_file)

            with open(os.path.abspath(DOCMAP_CACHE), mode="rb") as docmap_file:
                self.docmap = load(docmap_file)

            with open(os.path.abspath(TF_CACHE), mode="rb") as tf_file:
                self.term_frequencies = load(tf_file)
        except FileNotFoundError as e:
            raise

    def save(self):
        with open(os.path.abspath(INDEX_CACHE), mode="wb") as index_file:
            dump(self.index, index_file)

        with open(os.path.abspath(DOCMAP_CACHE), mode="wb") as docmap_file:
            dump(self.docmap, docmap_file)

        with open(os.path.abspath(TF_CACHE), mode="wb") as tf_file:
            dump(self.term_frequencies, tf_file)

    def get_tf(self, doc_id:int, term: str) -> int:
        return self.term_frequencies[doc_id].get(term,0)

    def get_idf(self, term: str):
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(term))
        return math.log((total_doc_count+1)/(term_match_doc_count+1))
def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build")
    load_parser = subparsers.add_parser("load")

    tf_parser = subparsers.add_parser("tf")
    tf_parser.add_argument("document_id", type=int)
    tf_parser.add_argument("term", type=str)

    idf_parser = subparsers.add_parser("idf")
    idf_parser.add_argument("term", type=str)

    tfidf_parser = subparsers.add_parser("tfidf")
    tfidf_parser.add_argument("document_id", type=int)
    tfidf_parser.add_argument("term", type=str)


    movies = load_movies(MOVIE_ADDRESS)
    stop_words = load_stop_words(STOPWORDS_ADDRESS)

    inverted_index = InvertedIndex(stop_words=stop_words)
    args = parser.parse_args()

    match args.command:
        case "search":
            query = args.query
            print(f"Searching for: {query}")

            query_tokens = tokenize_text(query, stop_words)

            result = []
            inverted_index.load()
            for token in query_tokens:
                if len(result) == 5:
                    break
                for movie_id in inverted_index.get_documents(token):
                    if len(result) == 5:
                        break
                    result.append(inverted_index.docmap[movie_id])

            result.sort(key=lambda movie: movie["id"])

            for movie in result:
                print(f"{movie["id"]}. {movie["title"]}")
        case "tf":
            inverted_index.load()
            document_id = args.document_id
            term = tokenize_term(args.term, stop_words)
            term_freq = inverted_index.get_tf(document_id, term)
            print(f"{term_freq}")
        case "idf":
            inverted_index.load()
            term = tokenize_term(args.term, stop_words)
            total_doc_count = len(inverted_index.docmap)
            print(total_doc_count)
            term_match_doc_count = len(inverted_index.get_documents(term))
            print(term_match_doc_count)
            idf = math.log((total_doc_count+1)/(term_match_doc_count+1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            inverted_index.load()
            document_id = args.document_id
            term = tokenize_term(args.term, stop_words)
            tf = inverted_index.get_tf(doc_id=document_id, term=term)
            idf = inverted_index.get_idf(term=term)
            tf_idf = tf * idf
            print(f"TF-IDF score of '{term}' in document '{document_id}': {tf_idf:.2f}")
        case "build":
            inverted_index.build()
            inverted_index.save()
        case "load":
            try:
                inverted_index.load()
            except Exception as e:
                print(f"{e}")

    

if __name__ == "__main__":
    main()
