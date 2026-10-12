import argparse
import json
import os 
import string 

from nltk.stem import PorterStemmer


def tokenize_text(word: str) -> list[str]:
    word = word.lower()  # case sensitivity
    without_punctuation = word.translate({ord(c): None for c in string.punctuation})  # punctuation
    tokens = without_punctuation.split()  # tokenization
    return tokens

def tokens_match(title_tokens: list[str], query_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

def remove_stop_words(stop_words: list[str], query_tokens: list[str]):
    query_set = set(query_tokens)
    return query_set.difference(stop_words)

def load_movies(file_path: str):
    with open(os.path.abspath(file_path)) as f:
        movies = json.load(f)["movies"]
    return movies

class InvertedIndex:
    def __init__(self) -> None: 
        self.index: dict[str, list[int]] = {}
        self.docmap: dict[int, dict[str,str]] = {}

    def __add_document(self, doc_id, text) -> None:
        pass

    def get_documents(self, term):
        pass

    def build(self):
        pass
    
def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    query = args.query
    
    movies = load_movies("data/movies.json")

    with open(os.path.abspath("data/stopwords.txt")) as f:
        stop_words = f.read().splitlines()


    stemmer = PorterStemmer()

    query_tokens = tokenize_text(query)
    query_processed = [stemmer.stem(token) for token in remove_stop_words(stop_words, query_tokens)]
    match args.command:
        case "search":
            print(f"Searching for: {query}")
            result = []
            for movie in movies:
                title_tokens = tokenize_text(movie["title"])
                match = tokens_match(title_tokens, query_processed)
                if match:
                    result.append(movie)
            
            result.sort(key=lambda movie: movie["id"])

            for index, movie in enumerate(result[:5], start=1):
                print(f"{index}. {movie["title"]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()