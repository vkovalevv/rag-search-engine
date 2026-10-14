import argparse
import math
import string

from config import settings
from lib.keyword_search import (
    BM25_B,
    BM25_K1,
    InvertedIndex,
    load_movies,
    load_stop_words,
    tokenize_term,
    tokenize_text,
)

# DATA_DIR = "/data"
MOVIE_ADDRESS = settings.MOVIE_ADDRESS
STOPWORDS_ADDRESS = settings.STOPWORDS_ADDRESS

# CACHE_DIR = "/cache"
INDEX_CACHE = settings.INDEX_CACHE
DOCMAP_CACHE = settings.DOCMAP_CACHE
TF_CACHE = settings.TF_CACHE
DOCLENGTHS_CACHE = settings.DOCLENGTHS_CACHE


def tf_command(
    document_id: int, term: str, inverted_index: InvertedIndex, stop_words: set[str]
) -> float:
    inverted_index.load()
    term = tokenize_term(term, stop_words)
    term_freq = inverted_index.get_tf(document_id, term)
    return term_freq


def idf_command(
    term: str, inverted_index: InvertedIndex, stop_words: set[str]
) -> float:
    inverted_index.load()
    term = tokenize_term(term, stop_words)
    total_doc_count = len(inverted_index.docmap)
    term_match_doc_count = len(inverted_index.get_documents(term))
    idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
    return idf


def tf_idf_command(
    document_id: int, term: str, inverted_index: InvertedIndex, stop_words: set[str]
) -> float:
    inverted_index.load()
    term = tokenize_term(term, stop_words)
    tf = inverted_index.get_tf(doc_id=document_id, term=term)
    idf = inverted_index.get_idf(term=term)
    tf_idf = tf * idf
    return tf_idf


def bm25_idf_command(
    term: str, inverted_index: InvertedIndex, stop_words: set[str]
) -> float:
    inverted_index.load()
    term = tokenize_term(term, stop_words)
    bm25idf = inverted_index.get_bm25_idf(term)
    return bm25idf


def bm25_tf_command(
    document_id: int,
    term: str,
    inverted_index: InvertedIndex,
    stop_words: set[str],
    b: float = BM25_B,
) -> float:
    inverted_index.load()
    term = tokenize_term(term, stop_words)
    bm25_tf = inverted_index.get_bm25_tf(document_id, term)
    return bm25_tf

def bm25search_command(
    query: str,
    inverted_index: InvertedIndex, 
    limit,
):  
    inverted_index.load()
    bm25search_result = inverted_index.bm25_search(query,limit)
    return bm25search_result


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

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("document_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser(
    "bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, help="limit of result", default=5)

    movies = load_movies(MOVIE_ADDRESS)
    stop_words = load_stop_words(STOPWORDS_ADDRESS)

    inverted_index = InvertedIndex(stop_words, MOVIE_ADDRESS, INDEX_CACHE, DOCMAP_CACHE, TF_CACHE, DOCLENGTHS_CACHE)
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
            term = args.term
            term_freq = tf_command(document_id, term, inverted_index, stop_words)
            print(f"{term_freq}")
        case "idf":
            term = args.term
            idf = idf_command(term, inverted_index, stop_words)
            print(f"Inverse document frequency of '{term}': {idf:.2f}")
        case "tfidf":
            document_id = args.document_id
            term = args.term
            tf_idf = tf_idf_command(document_id, term, inverted_index, stop_words)
            print(f"TF-IDF score of '{term}' in document '{document_id}': {tf_idf:.2f}")
        case "bm25idf":
            term = args.term
            bm25idf = bm25_idf_command(term, inverted_index, stop_words)
            print(f"BM25 IDF score of '{term}': {bm25idf:.2f}")
        case "bm25tf":
            document_id = args.document_id
            term = args.term
            bm25tf = bm25_tf_command(document_id, term, inverted_index, stop_words)
            print(
                f"BM25 TF score of '{term}' in document '{document_id}': {bm25tf:.2f}"
            )
        case "bm25search":
            query = args.query
            limit = args.limit
            bm25search_result = bm25search_command(query, inverted_index, limit)
            for index, (document_id, score) in enumerate(bm25search_result, start=1):
                title = inverted_index.docmap[document_id]["title"]
                print(f"{index}. ({document_id}) {title} - Score: {score:.2f}")
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
