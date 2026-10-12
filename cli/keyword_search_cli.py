import argparse
import json
import os 
import string 

from pickle import dump, load
from collections import defaultdict
from nltk.stem import PorterStemmer

MOVIE_ADDRESS = "data/movies.json"
STOPWORDS_ADDRESS = "data/stopwords.txt"
stemmer = PorterStemmer()

def remove_stop_words(query_tokens: list[str]):
    with open(os.path.abspath(STOPWORDS_ADDRESS)) as f:
        stop_words = f.read().splitlines()
    query_set = set(query_tokens)
    return query_set.difference(stop_words)

def tokenize_text(word: str) -> list[str]:
    word = word.lower()  # case sensitivity
    without_punctuation = word.translate({ord(c): None for c in string.punctuation})  # punctuation
    tokens_without_punctuation = without_punctuation.split()  # tokenization
    tokens_without_stopw = remove_stop_words(tokens_without_punctuation)
    tokens = [stemmer.stem(token) for token in tokens_without_stopw]
    return tokens

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
    def __init__(self) -> None: 
        self.index: dict[str, list[int]] = defaultdict(list)
        self.docmap: dict[int, dict] = {}

    def __add_document(self, doc_id, text) -> None:
        tokens = tokenize_text(text)
        for token in tokens:
            self.index[token].append(doc_id)      

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
            with open(os.path.abspath("cache/index.pkl"), mode="rb") as index_file:
                self.index = load(index_file)

            with open(os.path.abspath("cache/docmap.pkl"), mode="rb") as docmap_file:
                self.docmap = load(docmap_file)  
        except FileNotFoundError as e:
            raise

    def save(self):
        with open(os.path.abspath("cache/index.pkl"), mode="wb") as index_file:
            dump(self.index, index_file)

        with open(os.path.abspath("cache/docmap.pkl"), mode="wb") as docmap_file:
            dump(self.docmap, docmap_file)

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build")
    load_parser = subparsers.add_parser("load")
    try:
        args = parser.parse_args()
        query = args.query
    except AttributeError as e:
        query = ""
        print("Build")

    movies = load_movies(MOVIE_ADDRESS)

    query_tokens = tokenize_text(query)
    inverted_index = InvertedIndex()
    match args.command:
        case "search":
            print(f"Searching for: {query}")
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