import argparse

from lib.hybrid_search import HybridSearch, normalize_command, weighted_search_command, rrf_search_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser("normalize")
    normalize_parser.add_argument("scores", type=float, nargs="+")

    weighted_search_parser = subparser.add_parser("weighted-search")
    weighted_search_parser.add_argument("query", type=str)
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5)
    weighted_search_parser.add_argument("--limit", type=int, default=5)


    rrf_search_parser = subparser.add_parser("rrf-search")
    rrf_search_parser.add_argument("query", type=str)
    rrf_search_parser.add_argument("-k", type=int, default=60)
    rrf_search_parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    match args.command:
        case "help":
            parser.print_help()
        case "normalize":
            scores = args.scores
            normalize_scores = normalize_command(scores)
            for score in normalize_scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            query = args.query
            alpha = args.alpha
            limit = args.limit
            weighted_search_command(query, alpha, limit)
        case "rrf-search":
            query = args.query
            k = args.k
            limit = args.limit
            rrf_search_command(query, k, limit)


if __name__ == "__main__":
    main()