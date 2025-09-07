import json
from functools import cached_property
from pathlib import Path

from spacy.language import Language
from spacy.matcher import PhraseMatcher


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
    def matcher(self):
        """Register all aliases into a spaCy PhraseMatcher (cached)."""
        matcher = PhraseMatcher(self.nlp.vocab, attr="LEMMA")

        # collect all (canonical, alias) pairs
        canonical_aliases = [
            (canonical, alias)
            for entity_group in self.vocabulary.values()
            for canonical, aliases in entity_group.items()
            for alias in aliases
        ]

        # batch-process aliases with nlp.pipe
        alias_docs = list(self.nlp.pipe((alias for _, alias in canonical_aliases)))

        # group docs by canonical and add to matcher
        by_canonical = {}
        for (canonical, _), doc in zip(canonical_aliases, alias_docs):
            by_canonical.setdefault(canonical, []).append(doc)

        for canonical, docs in by_canonical.items():
            matcher.add(canonical, docs)

        return matcher

    @cached_property
    def kw_to_cat(self):
        """Map each canonical entry (e.g., 'movie') to its top-level category (e.g., 'type')."""
        return {
            keyword: category
            for category, keywords in self.vocabulary.items()
            for keyword in keywords
        }

    def extract_keywords(self, query: str):
        """Extract canonical keywords grouped by category from a query."""
        doc = self.nlp(query.lower())
        data: dict[str, set[str]] = {}

        for match_id, start, end in self.matcher(doc):
            canonical = self.nlp.vocab.strings[match_id]
            category = self.kw_to_cat.get(canonical)
            if category:
                data.setdefault(category, set()).add(canonical)

        return data