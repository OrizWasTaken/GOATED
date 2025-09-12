import json
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import DefaultDict, List, Dict

from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc


def load_json(file_path: str):
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
    """Loads a domain-specific vocabulary from JSON and manages it for query parsing."""

    def __init__(self, file_path: str, nlp: Language):
        self.nlp = nlp
        self.config = load_json(file_path)
        self.vocabulary = self.config["vocabulary"]

    @cached_property
    def matcher(self) -> PhraseMatcher:
        """Register all aliases into a spaCy PhraseMatcher (cached)."""
        matcher = PhraseMatcher(self.nlp.vocab, attr="LEMMA")

        # Collect aliases with their canonicals in one pass
        alias_canonical_pairs = [
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

    def extract_keywords(self, query: str):
        """Extract canonical keywords grouped by category from a query."""
        doc = self.nlp(query.lower())
        data: dict[str, set[str]] = {}

        for match_id, start, end in self.matcher(doc):
            canonical = self.nlp.vocab.strings[match_id]
            dimension = self.dim_by_canon_orth.get(match_id)
            if dimension:
                data.setdefault(dimension, set()).add(canonical)

        return data