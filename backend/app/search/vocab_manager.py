import json
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import DefaultDict, List, Dict, Set, Tuple, Optional

from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc


def load_json(file_path: str) -> dict:
    """Load and validate a JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f'"{path}" does not exist.')
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f'"{path}" is not a valid JSON file. {e}')


class DomainVocabManager:
    """
    Loads a domain-specific vocabulary from JSON and manages it for query parsing.

    Optimized to parse user queries and extract domain-specific
    dimensions with confidence scores.
    """

    # Valid token POS tags for domain terms
    VALID_POS_TAGS = {"NOUN", "ADJ", "PROPN", "NUM", "X", "ADV", "VERB"}

    def __init__(self, file_path: str, nlp: Language):
        """
        Initialize the DomainVocabManager.

        Args:
            file_path: Path to the domain vocabulary JSON file
            nlp: Initialized spaCy Language model
        """
        self.nlp = nlp
        self.config = load_json(file_path)
        self.vocabulary = self.config["vocabulary"]
        self.default_subject = self.config["default_subject"] # Required dimension, raise KeyError if missing.

        # Load dimension labels from config
        labels = self.config["dimension_labels"]
        self.SUBJECT = labels["SUBJECT"]  # Required dimension, raise KeyError if missing.
        self.CATEGORY = labels.get("CATEGORY", "category")
        self.SOURCE = labels.get("SOURCE", "source")
        self.TIMEFRAME = labels.get("TIMEFRAME", "timeframe")
        self.GEOGRAPHICAL = labels.get("GEOGRAPHICAL", "geographical")

        # Create reverse mapping for debugging
        self.dimension_names = {
            self.SUBJECT: "SUBJECT",
            self.CATEGORY: "CATEGORY",
            self.SOURCE: "SOURCE",
            self.TIMEFRAME: "TIMEFRAME",
            self.GEOGRAPHICAL: "GEOGRAPHICAL"
        }

    @cached_property
    def matcher(self) -> PhraseMatcher:
        """Register all aliases into a spaCy PhraseMatcher (cached)."""
        matcher = PhraseMatcher(self.nlp.vocab, attr="LEMMA")

        # Collect aliases with their canonicals in one pass
        alias_canonical_pairs: List[Tuple[str, str]] = [
            (alias, canonical)
            for entity_group in self.vocabulary.values()
            for canonical, aliases in entity_group.items()
            for alias in aliases
        ]

        # Early exit if no aliases
        if not alias_canonical_pairs:
            return matcher

        # Extract just aliases for batch processing
        alias_docs = list(self.nlp.pipe((alias for alias, _ in alias_canonical_pairs)))

        # Group docs by canonical in one pass
        by_canonical: DefaultDict[str, List[Doc]] = defaultdict(list)
        for (_, canonical), doc in zip(alias_canonical_pairs, alias_docs):
            by_canonical[canonical].append(doc)

        # Add to matcher - batch add if possible
        for canonical, docs in by_canonical.items():
            matcher.add(canonical, docs)

        return matcher

    @cached_property
    def dim_by_canon_orth(self) -> Dict[int, str]:
        """Map each canonical term's orth ID to its top-level dimension."""
        # Force matcher initialization to populate vocab with canonicals (IDs for PhraseMatcher patterns).
        _ = self.matcher

        return {
            self.nlp.vocab.strings.add(canonical): dimension
            for dimension, canonicals in self.vocabulary.items()
            for canonical in canonicals
        }

    def extract_keywords(self, query: str) -> Dict[str, any]:
        """
        Extract canonical keywords grouped by dimension from a query.

        Args:
            query: User query string (e.g., "best sci-fi thriller movies on Netflix")

        Returns:
            Dictionary with extracted dimensions and overall confidence score
        """
        # Process the query
        doc = self.nlp(query.lower().strip())

        # Find matches
        matches = self.matcher(doc)
        if not matches:
            # No matches found, return default structure
            return {
                self.SUBJECT: self.default_subject,
                self.CATEGORY: [],
                self.SOURCE: [],
                self.GEOGRAPHICAL: [],
                self.TIMEFRAME: [],
                "confidence": 0.0
            }

        # Initialize data structures
        dim_labels = [self.SUBJECT, self.CATEGORY, self.SOURCE, self.GEOGRAPHICAL, self.TIMEFRAME]
        data = {label: defaultdict(float) for label in dim_labels}

        # Process each match and calculate basic confidence
        for match_id, start, end in matches:
            span = doc[start:end]

            # Skip invalid POS tags
            if span.root.pos_ not in self.VALID_POS_TAGS:
                continue

            dimension = self.dim_by_canon_orth.get(match_id)
            if not dimension:
                continue

            canonical = self.nlp.vocab.strings[match_id]

            # Basic confidence (just 1.0 for now, will be enhanced later)
            confidence = 1.0

            data[dimension][canonical] = max(
                data[dimension][canonical],
                confidence
            )

        # Format final results with basic structure
        result = {
            self.SUBJECT: self._select_basic_subject(data),
            self.CATEGORY: list(data[self.CATEGORY].keys()),
            self.SOURCE: list(data[self.SOURCE].keys()),
            self.GEOGRAPHICAL: list(data[self.GEOGRAPHICAL].keys()),
            self.TIMEFRAME: list(data[self.TIMEFRAME].keys()),
            "confidence": 50.0 if any(data[label] for label in dim_labels) else 0.0
        }

        return result

    def _select_basic_subject(self, data: Dict[str, Dict[str, float]]) -> str:
        """Select the best subject or return default."""
        candidates = data[self.SUBJECT]

        if not candidates:
            return self.default_subject

        # Return the first match for now
        return next(iter(candidates))