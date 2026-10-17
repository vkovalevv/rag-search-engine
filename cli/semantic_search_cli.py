import argparse
from lib.semantic_search import chunking, search_command, semantic_chunking, verify_model, embed_text, verify_embeddings, embed_query_text

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command")

    verify_parser = subparsers.add_parser("verify")

    embed_parser = subparsers.add_parser("embed_text")
    embed_parser.add_argument("text", type=str)

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings")

    generate_embedding_parser = subparsers.add_parser("embed_query")
    generate_embedding_parser.add_argument("query", type=str)

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("--limit", type=int, default=5)

    chunk_parser = subparsers.add_parser("chunk")
    chunk_parser.add_argument("text", type=str)
    chunk_parser.add_argument("--chunk-size", type=int,default=200)
    chunk_parser.add_argument("--overlap", type=int)

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk")
    semantic_chunk_parser.add_argument("text", type=str)
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4)
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0)
    args = parser.parse_args()

    match args.command:
        case "--help":
            parser.print_help()
        case "verify":
            verify_model()
        case "embed_text":
            text = args.text
            embed_text(text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            query = args.query
            embed_query_text(query)
        case "chunk":
            text = args.text
            chunk_size = args.chunk_size
            overlap = args.overlap
            chunking(text, chunk_size, overlap)
        case "semantic_chunk":
            text = args.text
            max_chunk_size = args.max_chunk_size
            overlap = args.overlap
            semantic_chunking(text, max_chunk_size, overlap)
        case "search":
            query = args.query
            limit = args.limit
            search_command(query, limit)

if __name__ == "__main__":
    main()