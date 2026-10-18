from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env"
    )

    # DATA_DIR = "/data"
    MOVIE_ADDRESS: str = "data/movies.json"
    STOPWORDS_ADDRESS: str = "data/stopwords.txt"

    # CACHE_DIR = "/cache"
    INDEX_CACHE: str = "cache/index.pkl"
    DOCMAP_CACHE: str = "cache/docmap.pkl"
    TF_CACHE: str = "cache/tf.pkl"
    DOCLENGTHS_CACHE: str = "cache/doc_lengths.pkl"
    EMBEDDINGS_CACHE: str = "cache/movie_embeddings.npy"
    CHUNK_METADATA_CACHE:str = "cache/chunk_metadata.json"

    BM25_K1: float = 1.5
    BM25_B: float = 0.75

    TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"

settings = Settings()