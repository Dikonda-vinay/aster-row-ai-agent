"""Embedding-based retriever with document-precedence awareness.

Retrieval alone cannot know that a customer sentence "means" the same
thing as a policy paragraph beyond text similarity, so we deliberately
keep this simple and legible: embed each heading-level chunk once
(cached to disk), rank candidates by cosine similarity, and attach
metadata (status / policy_authority / supersedes) so the agent layer
can decide what may be cited as authoritative.

We do NOT silently drop superseded/draft content from retrieval — the
agent needs to be able to recognize and respond to a customer citing a
legacy or non-authoritative document (see the injection eval case). We
only refuse to let non-authoritative chunks stand in as the SOURCE for a
factual policy claim.
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app import config
from app.ingest import Chunk, load_documents


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, kb_dir: Path = config.KB_DIR):
        self.kb_dir = kb_dir
        self._model = None  # lazy-loaded, sentence-transformers is heavy to import
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None

    # -- model / index lifecycle -----------------------------------------
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        return self._model

    def _corpus_fingerprint(self) -> str:
        """Hash of file contents + embedding model name, to invalidate cache
        automatically when docs or the embedding model change."""
        h = hashlib.sha256()
        h.update(config.EMBEDDING_MODEL_NAME.encode())
        for path in sorted(self.kb_dir.glob("*.md")):
            h.update(path.name.encode())
            h.update(path.read_bytes())
        return h.hexdigest()

    def build_or_load_index(self, force_rebuild: bool = False) -> None:
        cache_path = config.INDEX_CACHE_PATH
        fingerprint = self._corpus_fingerprint()

        if not force_rebuild and cache_path.exists():
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("fingerprint") == fingerprint:
                self.chunks = cached["chunks"]
                self.embeddings = cached["embeddings"]
                return

        self.chunks = load_documents(self.kb_dir)
        texts = [c.text for c in self.chunks]
        vectors = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        self.embeddings = np.asarray(vectors, dtype=np.float32)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(
                {
                    "fingerprint": fingerprint,
                    "chunks": self.chunks,
                    "embeddings": self.embeddings,
                },
                f,
            )

    # -- search -----------------------------------------------------------
    def search(self, query: str, top_k: int = config.TOP_K) -> list[RetrievedChunk]:
        if self.embeddings is None:
            self.build_or_load_index()

        query_vec = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        scores = self.embeddings @ query_vec  # cosine sim, both sides normalized

        ranked_idx = np.argsort(-scores)[: max(top_k * 3, top_k)]
        results = [
            RetrievedChunk(chunk=self.chunks[i], score=float(scores[i]))
            for i in ranked_idx
        ]

        # Rerank: authoritative (active + official) chunks are promoted
        # above non-authoritative ones of similar relevance, but a
        # non-authoritative chunk is still surfaced if it's clearly what
        # the query is about (e.g. customer explicitly references it) —
        # we only apply a moderate boost, not exclusion.
        def sort_key(r: RetrievedChunk) -> float:
            boost = 0.05 if r.chunk.is_authoritative else 0.0
            return r.score + boost

        results.sort(key=sort_key, reverse=True)
        return results[:top_k]

    def relevant_results(self, query: str, top_k: int = config.TOP_K) -> list[RetrievedChunk]:
        """Search, then drop anything below the minimum relevance bar.

        An empty return means "the knowledge base doesn't appear to
        contain an answer to this" — the agent should abstain rather
        than force a citation onto a weak match.
        """
        results = self.search(query, top_k=top_k)
        return [r for r in results if r.score >= config.MIN_RELEVANCE_SCORE]
