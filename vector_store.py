"""
vector_store.py — a small wrapper class that hides Chroma behind two methods.

Why wrap Chroma instead of calling it directly everywhere?
    The agent and the ingestion script only ever need two operations:
    "put documents in" and "find documents similar to this question".
    By funneling both through this class, later upgrades (a reranker,
    a paid embedding model, a second data source) only touch THIS file —
    the agent and UI never notice.

How the embedding works (important for understanding RAG):
    When you call add_documents(), Chroma automatically converts each text
    into an "embedding" — a list of ~384 numbers that captures its meaning —
    using a small model (all-MiniLM-L6-v2) that runs locally and free.
    When you call query(), your question gets embedded the same way, and
    Chroma returns the stored chunks whose numbers are closest to the
    question's numbers. "Closest numbers" ≈ "most similar meaning".

Caching: Chroma persists everything to data/chroma/ on disk, so embeddings
    are computed exactly once. Restarting the app costs nothing.
"""

import chromadb
import re
from nltk.stem import PorterStemmer

from config import CHROMA_DIR, COLLECTION_NAME, TOP_K, FETCH_K, RERANK_ENABLED, RERANK_MODEL, HYBRID_SEARCH_ENABLED

_stemmer = PorterStemmer()


def _tokenize(text: str) -> list[str]:
    """
    Lowercase, STEMMED word tokens for BM25. Stemming reduces related word
    forms to one root (sensor/sensors -> "sensor", program/programs ->
    "program"), so a plural question still matches a singular corpus
    mention and vice versa — otherwise BM25's exact-token matching treats
    them as unrelated words and silently misses an obviously relevant chunk.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [_stemmer.stem(w) for w in words]


def _matches_where(meta: dict, where: dict) -> bool:
    """Minimal Chroma-style `where` matcher for filtering BM25's in-memory
    doc list (Chroma's own `where` only applies to the embedding side)."""
    for key, condition in where.items():
        value = meta.get(key)
        if isinstance(condition, dict) and "$in" in condition:
            if value not in condition["$in"]:
                return False
        elif value != condition:
            return False
    return True


class VectorStore:
    """Thin interface over a persistent Chroma collection."""

    def __init__(self, collection_name: str = COLLECTION_NAME):
        # PersistentClient = "save to disk", as opposed to in-memory only.
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        # get_or_create: reuses the existing collection if ingestion already
        # ran (that's our embedding cache), creates it empty otherwise.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            # cosine distance is the standard way to compare embeddings
            metadata={"hnsw:space": "cosine"},
        )
        self._reranker = None   # loaded on first query (keeps startup fast)
        self._bm25 = None          # lazy-built BM25 index (see _get_bm25)
        self._bm25_docs = None     # parallel [{"id","text","metadata"}] for BM25

    def count(self) -> int:
        """How many chunks are stored? (0 means ingestion hasn't run.)"""
        return self.collection.count()

    def _get_reranker(self):
        """Load the FlashRank reranker once, on first use."""
        if self._reranker is None:
            from flashrank import Ranker
            self._reranker = Ranker(model_name=RERANK_MODEL)
        return self._reranker

    def _get_bm25(self):
        """
        Build (once, lazily) a keyword-search index over every document in
        this collection. BM25 catches exact-term matches that embeddings
        can miss — e.g. a broad question ("tell me everything about
        sensors") where the QUESTION's wording doesn't resemble the
        corpus's technical prose, but the literal keyword IS present.
        """
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            raw = self.collection.get(include=["documents", "metadatas"])
            self._bm25_docs = [
                {"id": id_, "text": text, "metadata": meta}
                for id_, text, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])
            ]
            self._bm25 = BM25Okapi([_tokenize(d["text"]) for d in self._bm25_docs])
        return self._bm25

    def _bm25_search(self, query_text: str, n: int, where: dict | None = None) -> list[dict]:
        bm25 = self._get_bm25()
        scores = bm25.get_scores(_tokenize(query_text))
        indices = range(len(scores))
        if where:
            # Filter to matching docs BEFORE taking the top n, not after —
            # otherwise a year filter could lose slots to the wrong year's
            # docs before we ever get to check them.
            indices = [i for i in indices if _matches_where(self._bm25_docs[i]["metadata"], where)]
        ranked = sorted(indices, key=lambda i: scores[i], reverse=True)[:n]
        return [
            {"id": self._bm25_docs[i]["id"], "text": self._bm25_docs[i]["text"],
             "metadata": self._bm25_docs[i]["metadata"], "distance": None}
            for i in ranked if scores[i] > 0   # skip zero-overlap results
        ]

    def add_documents(self, texts: list[str], metadatas: list[dict], ids: list[str]):
        """
        Store chunks. Each chunk needs:
          - text: the chunk's content (gets embedded automatically)
          - metadata: dict with program name + page numbers (for citations!)
          - id: a stable unique string, so re-adding overwrites, not duplicates
        """
        self.collection.add(documents=texts, metadatas=metadatas, ids=ids)
        self._bm25 = None   # invalidate the cached index — new docs exist now

    def has_metadata_value(self, field: str, value) -> bool:
        """Cheap existence check, e.g. "has this fiscal_year already been
        ingested?" — used by multi-year ingest scripts as their per-year
        cache check (the collection-wide store.count() check isn't enough
        once a collection holds more than one year)."""
        result = self.collection.get(where={field: value}, limit=1)
        return len(result["ids"]) > 0

    def query(self, query_text: str, top_k: int = TOP_K, where: dict | None = None) -> list[dict]:
        """
        Retrieval pipeline (each stage toggleable in config):
          1. Embedding search  — always runs; catches semantic/meaning matches.
          2. BM25 keyword search — merged in if HYBRID_SEARCH_ENABLED; catches
             exact-term matches embeddings miss on broad/vague questions.
          3. Reranker — re-scores the merged shortlist, if RERANK_ENABLED.
        `where`, e.g. {"fiscal_year": {"$in": ["FY2024", "FY2025"]}}, scopes
        retrieval to a subset of a multi-year collection.
        Each hit: {"id", "text", "metadata", "distance", "rerank_score"?}.
        """
        n_fetch = FETCH_K if RERANK_ENABLED else top_k

        query_kwargs = {"query_texts": [query_text], "n_results": n_fetch}
        if where:
            query_kwargs["where"] = where
        results = self.collection.query(**query_kwargs)
        hits, seen_ids = [], set()
        for id_, text, meta, dist in zip(
            results["ids"][0], results["documents"][0],
            results["metadatas"][0], results["distances"][0],
        ):
            hits.append({"id": id_, "text": text, "metadata": meta, "distance": dist})
            seen_ids.add(id_)

        if HYBRID_SEARCH_ENABLED:
            for h in self._bm25_search(query_text, n=n_fetch, where=where):
                if h["id"] not in seen_ids:
                    hits.append(h)
                    seen_ids.add(h["id"])

        if RERANK_ENABLED and hits:
            from flashrank import RerankRequest
            reranker = self._get_reranker()
            passages = [{"id": i, "text": h["text"]} for i, h in enumerate(hits)]
            ranked = reranker.rerank(RerankRequest(query=query_text, passages=passages))
            reordered = []
            for r in ranked:
                h = hits[r["id"]]
                h["rerank_score"] = float(r["score"])
                reordered.append(h)
            hits = reordered

        return hits[:top_k]
    
    def reset(self):
        """Delete and recreate the collection (used by ingest --rebuild)."""
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )
