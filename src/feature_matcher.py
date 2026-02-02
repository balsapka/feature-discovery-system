"""
Feature Matcher - Matches generated features against an internal feature bank.

Supports multiple matching backends:
- In-memory vector search (default, no external dependencies)
- Fuzzy string matching
- TF-IDF
- Hybrid (fuzzy + semantic)
- Qdrant (for large-scale production)
"""
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol

from .models.schemas import Feature

logger = logging.getLogger(__name__)


@dataclass
class FeatureMatch:
    """A single match result from the internal feature bank."""
    internal_name: str
    similarity_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureMatchResult:
    """Match results for a single generated feature."""
    generated_feature: str
    generated_description: str
    matches: List[FeatureMatch] = field(default_factory=list)

    @property
    def top_match(self) -> Optional[FeatureMatch]:
        """Return the best match, if any."""
        return self.matches[0] if self.matches else None

    def has_strong_match(self, threshold: float = 0.8) -> bool:
        """Check if there's a strong match above threshold."""
        return self.top_match is not None and self.top_match.similarity_score >= threshold


class Embedder(Protocol):
    """Protocol for embedding models."""

    def encode(
        self, texts: List[str], normalize_embeddings: bool = True
    ) -> Any:
        """Encode texts to vectors."""


class BaseFeatureMatcher(ABC):
    """Abstract base class for feature matchers."""

    def __init__(self, top_k: int = 5):
        """
        Initialize the matcher.

        Args:
            top_k: Number of top matches to return per feature
        """
        self.top_k = top_k

    @abstractmethod
    def index(self, features: List[Dict[str, Any]]) -> None:
        """
        Index the internal feature bank.

        Args:
            features: List of feature dicts with at least 'name' and 'description' keys
        """

    @abstractmethod
    def search(self, query: str) -> List[FeatureMatch]:
        """
        Search for similar features.

        Args:
            query: Query text (typically "name: description")

        Returns:
            List of FeatureMatch objects
        """

    def match_feature(self, feature: Feature) -> FeatureMatchResult:
        """
        Match a single generated feature against the internal bank.

        Args:
            feature: Generated Feature object

        Returns:
            FeatureMatchResult with matches
        """
        query = f"{feature.name}: {feature.description}"
        matches = self.search(query)

        return FeatureMatchResult(
            generated_feature=feature.name,
            generated_description=feature.description,
            matches=matches
        )

    def match_all(
        self,
        features: List[Feature],
        parallel: bool = True,
        max_workers: int = 5
    ) -> List[FeatureMatchResult]:
        """
        Match all generated features against the internal bank.

        Args:
            features: List of generated Feature objects
            parallel: Whether to run searches in parallel
            max_workers: Number of parallel workers

        Returns:
            List of FeatureMatchResult objects
        """
        if parallel and len(features) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(self.match_feature, features))
            return results

        return [self.match_feature(f) for f in features]


class InMemoryVectorMatcher(BaseFeatureMatcher):
    """
    In-memory vector search using sentence-transformers.

    Good for feature banks up to ~100K features.
    No external infrastructure required.
    """

    def __init__(
        self,
        top_k: int = 5,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the in-memory vector matcher.

        Args:
            top_k: Number of top matches to return
            model_name: SentenceTransformer model name
        """
        super().__init__(top_k)
        self.model_name = model_name
        self.model = None
        self.vectors = None
        self.features = []

    def _load_model(self):
        """Lazy load the embedding model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                logger.info("Loaded embedding model: %s", self.model_name)
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for InMemoryVectorMatcher. "
                    "Install with: pip install sentence-transformers"
                ) from exc

    def index(self, features: List[Dict[str, Any]]) -> None:
        """Index the internal feature bank."""
        self._load_model()

        self.features = features
        texts = [
            f"{f.get('name', '')}: {f.get('description', '')}"
            for f in features
        ]

        logger.info("Indexing %d features...", len(features))
        self.vectors = self.model.encode(texts, normalize_embeddings=True)
        logger.info("Indexing complete. Vector shape: %s", self.vectors.shape)

    def search(self, query: str) -> List[FeatureMatch]:
        """Search for similar features using cosine similarity."""
        if self.vectors is None:
            raise RuntimeError("Index not built. Call index() first.")

        self._load_model()

        import numpy as np

        query_vec = self.model.encode(query, normalize_embeddings=True)
        scores = self.vectors @ query_vec  # Cosine similarity (normalized)

        top_indices = np.argsort(scores)[-self.top_k:][::-1]

        return [
            FeatureMatch(
                internal_name=self.features[i].get('name', ''),
                similarity_score=float(scores[i]),
                metadata=self.features[i]
            )
            for i in top_indices
            if scores[i] > 0  # Filter out zero/negative scores
        ]


class FuzzyMatcher(BaseFeatureMatcher):
    """
    Fuzzy string matching using rapidfuzz.

    Fast and good for catching naming variations.
    No embeddings required.
    """

    def __init__(self, top_k: int = 5, threshold: int = 50):
        """
        Initialize the fuzzy matcher.

        Args:
            top_k: Number of top matches to return
            threshold: Minimum fuzzy score (0-100) to include
        """
        super().__init__(top_k)
        self.threshold = threshold
        self.features = []
        self.name_to_feature = {}

    def index(self, features: List[Dict[str, Any]]) -> None:
        """Index the internal feature bank."""
        self.features = features
        self.name_to_feature = {f.get('name', ''): f for f in features}
        logger.info("Indexed %d features for fuzzy matching", len(features))

    def search(self, query: str) -> List[FeatureMatch]:
        """Search using fuzzy string matching on feature names."""
        try:
            from rapidfuzz import fuzz, process
        except ImportError as exc:
            raise ImportError(
                "rapidfuzz is required for FuzzyMatcher. "
                "Install with: pip install rapidfuzz"
            ) from exc

        # Extract just the name part from "name: description"
        query_name = query.split(":")[0].strip()

        internal_names = list(self.name_to_feature.keys())
        matches = process.extract(
            query_name,
            internal_names,
            scorer=fuzz.ratio,
            limit=self.top_k
        )

        return [
            FeatureMatch(
                internal_name=name,
                similarity_score=score / 100.0,  # Normalize to 0-1
                metadata=self.name_to_feature.get(name, {})
            )
            for name, score, _ in matches
            if score >= self.threshold
        ]


class TFIDFMatcher(BaseFeatureMatcher):
    """
    TF-IDF based matching using sklearn.

    Lightweight, no neural network required.
    Good baseline for keyword-based matching.
    """

    def __init__(self, top_k: int = 5):
        """Initialize the TF-IDF matcher."""
        super().__init__(top_k)
        self.vectorizer = None
        self.matrix = None
        self.features = []

    def index(self, features: List[Dict[str, Any]]) -> None:
        """Index the internal feature bank."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for TFIDFMatcher. "
                "Install with: pip install scikit-learn"
            ) from exc

        self.features = features
        texts = [
            f"{f.get('name', '')} {f.get('description', '')}"
            for f in features
        ]

        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(texts)
        logger.info("Indexed %d features with TF-IDF", len(features))

    def search(self, query: str) -> List[FeatureMatch]:
        """Search using TF-IDF cosine similarity."""
        if self.matrix is None:
            raise RuntimeError("Index not built. Call index() first.")

        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]

        top_indices = np.argsort(scores)[-self.top_k:][::-1]

        return [
            FeatureMatch(
                internal_name=self.features[i].get('name', ''),
                similarity_score=float(scores[i]),
                metadata=self.features[i]
            )
            for i in top_indices
            if scores[i] > 0
        ]


class HybridMatcher(BaseFeatureMatcher):
    """
    Hybrid matcher combining fuzzy string matching and semantic search.

    Best accuracy: catches both naming variations and semantic equivalents.
    """

    def __init__(
        self,
        top_k: int = 5,
        fuzzy_boost: float = 1.2,
        fuzzy_threshold: int = 50,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the hybrid matcher.

        Args:
            top_k: Number of top matches to return
            fuzzy_boost: Multiplier for scores that also match fuzzy
            fuzzy_threshold: Minimum fuzzy score to count as a fuzzy match
            model_name: SentenceTransformer model name
        """
        super().__init__(top_k)
        self.fuzzy_boost = fuzzy_boost
        self.vector_matcher = InMemoryVectorMatcher(
            top_k=top_k * 2,  # Get more candidates for reranking
            model_name=model_name
        )
        self.fuzzy_matcher = FuzzyMatcher(
            top_k=top_k * 2,
            threshold=fuzzy_threshold
        )
        self.features = []

    def index(self, features: List[Dict[str, Any]]) -> None:
        """Index the internal feature bank in both matchers."""
        self.features = features
        self.vector_matcher.index(features)
        self.fuzzy_matcher.index(features)
        logger.info("Indexed %d features in hybrid matcher", len(features))

    def search(self, query: str) -> List[FeatureMatch]:
        """Search using both fuzzy and semantic, then combine results."""
        # Get fuzzy matches (for boosting)
        fuzzy_results = self.fuzzy_matcher.search(query)
        fuzzy_names = {m.internal_name for m in fuzzy_results}

        # Get semantic matches
        semantic_results = self.vector_matcher.search(query)

        # Combine scores, boosting fuzzy matches
        combined = {}
        for match in semantic_results:
            score = match.similarity_score
            if match.internal_name in fuzzy_names:
                score *= self.fuzzy_boost
            combined[match.internal_name] = FeatureMatch(
                internal_name=match.internal_name,
                similarity_score=min(score, 1.0),  # Cap at 1.0
                metadata=match.metadata
            )

        # Add fuzzy-only matches that weren't in semantic results
        for match in fuzzy_results:
            if match.internal_name not in combined:
                combined[match.internal_name] = match

        # Sort by score and return top_k
        sorted_matches = sorted(
            combined.values(),
            key=lambda m: m.similarity_score,
            reverse=True
        )
        return sorted_matches[:self.top_k]


class QdrantMatcher(BaseFeatureMatcher):  # pylint: disable=too-many-instance-attributes
    """
    Qdrant-based matcher for large-scale production use.

    Requires running Qdrant instance.
    Best for >100K features.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        top_k: int = 5,
        qdrant_url: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "internal_features",
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the Qdrant matcher.

        Args:
            top_k: Number of top matches to return
            qdrant_url: Qdrant server URL
            qdrant_port: Qdrant server port
            collection_name: Name of the Qdrant collection
            model_name: SentenceTransformer model name
        """
        super().__init__(top_k)
        self.qdrant_url = qdrant_url
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        self.model_name = model_name
        self.client = None
        self.model = None

    def _load_dependencies(self):
        """Lazy load Qdrant client and embedding model."""
        if self.client is None:
            try:
                from qdrant_client import QdrantClient
                self.client = QdrantClient(
                    host=self.qdrant_url,
                    port=self.qdrant_port
                )
            except ImportError as exc:
                raise ImportError(
                    "qdrant-client is required for QdrantMatcher. "
                    "Install with: pip install qdrant-client"
                ) from exc

        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for QdrantMatcher. "
                    "Install with: pip install sentence-transformers"
                ) from exc

    def index(self, features: List[Dict[str, Any]]) -> None:
        """Index features into Qdrant collection."""
        self._load_dependencies()

        from qdrant_client.models import Distance, VectorParams, PointStruct

        texts = [
            f"{f.get('name', '')}: {f.get('description', '')}"
            for f in features
        ]
        vectors = self.model.encode(texts, normalize_embeddings=True)

        # Create collection if needed
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vectors.shape[1],
                    distance=Distance.COSINE
                )
            )

        # Upsert points
        points = [
            PointStruct(
                id=i,
                vector=vectors[i].tolist(),
                payload=features[i]
            )
            for i in range(len(features))
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info("Indexed %d features into Qdrant collection '%s'",
                    len(features), self.collection_name)

    def search(self, query: str) -> List[FeatureMatch]:
        """Search Qdrant for similar features."""
        self._load_dependencies()

        query_vec = self.model.encode(query, normalize_embeddings=True).tolist()

        # pylint: disable=no-member
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=self.top_k
        ).points

        return [
            FeatureMatch(
                internal_name=hit.payload.get('name', ''),
                similarity_score=hit.score,
                metadata=hit.payload
            )
            for hit in results
        ]


# Factory function for easy instantiation
def create_matcher(
    backend: str = "vector",
    top_k: int = 5,
    **kwargs
) -> BaseFeatureMatcher:
    """
    Factory function to create a feature matcher.

    Args:
        backend: One of "vector", "fuzzy", "tfidf", "hybrid", "qdrant"
        top_k: Number of top matches to return
        **kwargs: Additional arguments for the specific matcher

    Returns:
        Configured BaseFeatureMatcher instance
    """
    matchers = {
        "vector": InMemoryVectorMatcher,
        "fuzzy": FuzzyMatcher,
        "tfidf": TFIDFMatcher,
        "hybrid": HybridMatcher,
        "qdrant": QdrantMatcher,
    }

    if backend not in matchers:
        raise ValueError(
            f"Unknown backend '{backend}'. Choose from: {list(matchers.keys())}"
        )

    return matchers[backend](top_k=top_k, **kwargs)
