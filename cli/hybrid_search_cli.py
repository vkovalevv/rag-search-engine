import argparse
from openai import OpenAI

from config import settings
from lib.hybrid_search import (
    HybridSearch,
    normalize_command,
    weighted_search_command,
    rrf_search_command,
)


def enhance_query(client: OpenAI, query: str) -> str:
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": f"""Fix any spelling errors in the user-provided movie search query below.
                Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                Preserve punctuation and capitalization unless a change is required for a typo fix.
                If there are no spelling errors, or if you're unsure, output the original query unchanged.
                Output only the final query text, nothing else.
                User query: "{query}"
                """,
            }
        ],
    )
    enhanced_query = response.choices[0].message.content
    return enhanced_query


def expand_query(client: OpenAI, query: str) -> str:
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": f"""Expand the user-provided movie search query below with related terms.
                Add synonyms and related concepts that might appear in movie descriptions.
                Keep expansions relevant and focused.
                Output only the additional terms; they will be appended to the original query.

                Examples:
                - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                - "action movie with bear" -> "action thriller bear chase fight adventure"
                - "comedy with bear" -> "comedy funny bear humor lighthearted"

                User query: "{query}"
                """,
            }
        ],
    )
    enhanced_query = response.choices[0].message.content
    return enhanced_query


def rewrite_query(client: OpenAI, query: str) -> str:
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": f"""Rewrite the user-provided movie search query below to be more specific and searchable.

                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions (horror = scary, animation = cartoon)
                - Keep the rewritten query concise (under 10 words)
                - It should be a Google-style search query, specific enough to yield relevant results
                - Don't use boolean logic

                Examples:
                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                If you cannot improve the query, output the original unchanged.
                Output only the rewritten query text, nothing else.

                User query: "{query}"
                """,
            }
        ],
    )
    enhanced_query = response.choices[0].message.content
    return enhanced_query


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
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    api_key = settings.OPENROUTER_API_KEY

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

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
            enhance = args.enhance
            enhanced_query = query
            match enhance:
                case "spell":
                    enhanced_query = enhance_query(client, query)
                case "expand":
                    enhanced_query = expand_query(client, query)
                case "rewrite":
                    enhanced_query = rewrite_query(client, query)
                    
            print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
            rrf_search_command(enhanced_query, k, limit)


if __name__ == "__main__":
    main()
