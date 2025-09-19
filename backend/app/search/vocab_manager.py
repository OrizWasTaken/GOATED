import re
import json
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Any

from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Token, Doc, Span


def load_json(file_path: str) -> dict[str, str | list[str] | dict[str, Any]]:
    """Load and validate a JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f'"{path}" does not exist.')
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f'"{path}" is not a valid JSON file. {e}')


# Load mappings and build lookup tables
def build_lookup(mappings: dict[str, Any]) -> dict[str, str]:
    return {
        term.lower(): key
        for key, terms in mappings.items()
        if isinstance(terms, list)
        for term in terms
    }

country_mappings = load_json(r"vocab/country_mappings.json")
timeline_mappings = load_json(r"vocab/timeline_mappings.json")

country_lookup = build_lookup(country_mappings)
timeline_lookup = build_lookup(timeline_mappings)

# Compile regex patterns
def create_pattern(lookup: dict[str, str]) -> re.Pattern[str]:
    return re.compile(
        r'\b(' + '|'.join(re.escape(term) for term in lookup.keys()) + r')\b',
        re.IGNORECASE
    )

country_pattern = create_pattern(country_lookup)
timeline_lookup = create_pattern(timeline_lookup)

year_pattern = re.compile(
    r'\b(?:(?:19|20)\d{2}\'?s|\d{2}\'?s|(?:19|20)\d{2})\b',
    re.IGNORECASE
)

def extract_regions(query: str) -> list[str]:
    """Extract all country/region mentions from text."""
    return [country_lookup[match.lower()] for match in country_pattern.findall(query)]

def extract_timelines(query: str) -> list[str]:
    """Extract all date mentions from text."""
    return year_pattern.findall(query) + timeline_lookup.findall(query)

class DomainVocabManager:
    """
    Loads a domain-specific vocabulary from JSON and manages it for query parsing.

    Optimized to parse user queries and extract domain-specific
    dimensions with confidence scores.
    """

    # Valid token POS tags for domain terms
    VALID_POS_TAGS = {"NOUN", "ADJ", "PROPN", "NUM", "X", "ADV", "VERB"}

    # Dependency confidence maps optimized for movie domain
    dep_confidences = {
        "SUBJECT": {
            "ROOT": 1.5,  # Root word
            "dobj": 1.2,  # Direct object (e.g., "best movies")
            "nsubj": 0.8,  # Nominal subject
            "pobj": 0.2,  # Object of preposition (e.g., "for movies")
            "attr": 0.15,  # Attribute
        },
        "CATEGORY": {
            "conj": 0.5,  # Conjunction (e.g., "action and thriller")
            "appos": 0.4  # Apposition
        },
        "SOURCE": {
            "pobj": 1.0,  # Object of preposition (e.g., "on Netflix")
            "prep": 0.8,  # Prepositional modifier
            "nmod": 0.6,  # Nominal modifier
            "conj": 0.4  # Conjunction (e.g., "Netflix and Hulu")
        },
    }

    # Confidence thresholds for dimension selection
    thresholds = {
        "SUBJECT": 2.0,  # Minimum: Tag analysis & reasonable dependency pattern.
        "CATEGORY": 2.0,  # Minimum: POS patterns & context analysis.
        "SOURCE": 2.5,  # High emphasis on domain's vocabulary.
    }

    # Weights for overall confidence calculation
    dimension_weights = {
        "SUBJECT": 0.45,  # Subject (movie/book) is very important (40%)
        "CATEGORY": 0.2,  # Category (thriller/non-fiction) is quite important (20%)
        "SOURCE": 0.2,  # Source (Netflix, Udemy) is quite important (20%)
        "GEOGRAPHICAL": 0.15,  # Geographical hints (k-drama, hollywood) can be important (15%)
    }

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
        self.default_subject = self.config["default_subject"]

        # Load dimension labels from config
        labels = self.config["dimension_labels"] # Required key, raise KeyError if missing.
        self.SUBJECT = labels["SUBJECT"]  # Required dimension, raise KeyError if missing.
        self.CATEGORY = labels.get("CATEGORY", "category")
        self.SOURCE = labels.get("SOURCE", "source")
        self.GEOGRAPHICAL = labels.get("GEOGRAPHICAL", "geographical")

        # Create reverse mapping for debugging
        self.dimension_names = {
            self.SUBJECT: "SUBJECT",
            self.CATEGORY: "CATEGORY",
            self.SOURCE: "SOURCE",
            self.GEOGRAPHICAL: "GEOGRAPHICAL"
        }

    @cached_property
    def matcher(self) -> PhraseMatcher:
        """Register all aliases into a spaCy PhraseMatcher (cached)."""
        matcher = PhraseMatcher(self.nlp.vocab, attr="LEMMA")

        # Collect aliases with their canonicals in one pass
        alias_canonical_pairs = []
        for dimension, entity_group in self.vocabulary.items():
            for canonical, aliases in entity_group.items():
                for alias in aliases:
                    alias_canonical_pairs.append((alias, canonical))

        # Early exit if no aliases
        if not alias_canonical_pairs:
            return matcher

        # Process aliases in batch for efficiency
        alias_texts = [alias for alias, _ in alias_canonical_pairs]
        alias_docs = list(self.nlp.pipe(alias_texts, disable=["ner", "parser"]))

        # Group docs by canonical
        by_canonical = defaultdict(list)
        for (_, canonical), doc in zip(alias_canonical_pairs, alias_docs):
            by_canonical[canonical].append(doc)

        # Add patterns to matcher
        for canonical, docs in by_canonical.items():
            matcher.add(canonical, docs)

        return matcher

    @cached_property
    def dim_by_canon_orth(self) -> dict[int, str]:
        """Map each canonical term's orth ID to its dimension."""
        _ = self.matcher  # Force matcher initialization

        mapping = {}
        for dimension, canonicals in self.vocabulary.items():
            for canonical in canonicals:
                orth_id = self.nlp.vocab.strings.add(canonical)
                mapping[orth_id] = dimension

        return mapping

    def extract_keywords(self, query: str) -> dict[str, str | float | list[str]]:
        """
        Extract canonical keywords grouped by dimension from a query.

        Args:
            query: User query string (e.g., "best sci-fi thriller movies on Netflix")

        Returns:
            dictionary with extracted dimensions and overall confidence score
        """
        # Create local aliases for dimension labels.
        subject = self.SUBJECT
        category = self.CATEGORY
        source = self.SOURCE
        geographical = self.GEOGRAPHICAL

        # Process the query
        query = query.lower().strip()
        doc = self.nlp(query)

        # Find matches
        matches = self.matcher(doc)
        if not matches:
            # No matches found, return default structure
            return {
                subject: self.default_subject,
                category: [],
                source: [],
                geographical: [],
                "timeframe": [],
                "confidence": 0.0
            }

        # Initialize data structures
        dim_labels = (subject, category, source, geographical)
        data = {label: defaultdict(float) for label in dim_labels}
        confidences_dict = {}

        # Pre-compute mappings for efficiency
        dim_by_tok = self._dim_by_tok(matches)
        ent_label_by_tok = self._ent_label_by_tok(doc)

        # Process each match
        for match in matches:
            self._process_match(doc, match, data, dim_by_tok, ent_label_by_tok)

        # Format final results
        result = {
            subject: self._select_subject(subject, data, confidences_dict),
            category: self._filter_dimension(category, data, confidences_dict),
            source: self._filter_dimension(source, data, confidences_dict),
            geographical: self._filter_dimension(geographical, data, confidences_dict) + extract_regions(query),
            "timeframe": extract_timelines(query),
            "confidence": self._calculate_overall_confidence(confidences_dict, doc)
        }

        return result

    def _process_match(
            self, doc: Doc,
            match: tuple[int, int, int],
            data: dict[str, dict[str, float]],
            dim_by_tok: dict[int, set[str]],
            ent_label_by_tok: dict[int, set[str]]
    ) -> None:
        """
        Process a single vocabulary match and update confidence data.

        Validates the match, calculates confidence scores across all dimensions,
        and updates the data structure. A single canonical term can contribute
        confidence to multiple dimensions based on linguistic context analysis.

        Args:
            doc: The spaCy Doc object containing the query
            match: tuple of (match_id, start, end) from PhraseMatcher
            data: dictionary mapping dimension labels to canonical terms and confidences
            dim_by_tok: Mapping of token indices to their matched dimensions
            ent_label_by_tok: Mapping of token indices to their spaCy entity labels
        """
        match_id, start, end = match
        span = doc[start:end]

        # Skip invalid POS tags
        if span.root.pos_ not in self.VALID_POS_TAGS:
            return None

        dimension = self.dim_by_canon_orth.get(match_id)
        if not dimension:
            return None

        canonical = self.nlp.vocab.strings[match_id]

        # Calculate confidences for all dimensions
        confidences = self._calculate_all_confidences(
            span, doc, dim_by_tok, ent_label_by_tok
        )

        # Update maximum confidence for each dimension
        for dim_label, confidence in confidences.items():
            if confidence > 0:
                data[dim_label][canonical] = max(
                    data[dim_label][canonical],
                    confidence
                )

        return None

    def _dim_by_tok(self, matches: list[tuple[int, int, int]]) -> dict[int, set[str]]:
        """Map token indices to their dimensions."""
        dim_by_tok = defaultdict(set)

        for match_id, start, end in matches:
            dimension = self.dim_by_canon_orth.get(match_id)
            if dimension:
                for i in range(start, end):
                    dim_by_tok[i].add(dimension)

        return dict(dim_by_tok)

    @staticmethod
    def _ent_label_by_tok(doc: Doc) -> dict[int, set[str]]:
        """Map token indices to their spaCy entity labels."""
        entities = doc.ents

        # Early exit if no entities.
        if not entities:
            return {}

        ent_by_tok = defaultdict(set)
        for ent in entities:
            for i in range(ent.start, ent.end):
                ent_by_tok[i].add(ent.label_)

        return dict(ent_by_tok)

    def _calculate_all_confidences(
            self, span: Span, doc: Doc,
            dim_by_tok: dict[int, set[str]],
            ent_label_by_tok: dict[int, set[str]]
    ) -> dict[str, float]:
        """Calculate confidence scores for all dimensions."""
        root = span.root

        # Get surrounding tokens within a window size of 1.
        start = max(0, span.start - 1)
        end = min(len(doc), span.end + 2)
        context_tokens =  doc[start:end]

        # Analyze context
        context_dims = set()
        for token in context_tokens:
            if token.i != root.i:  # Exclude the root token itself
                context_dims.update(dim_by_tok.get(token.i, []))

        return {
            self.SUBJECT: self._calc_subject_confidence(root, dim_by_tok, context_dims),
            self.CATEGORY: self._calc_category_confidence(root, dim_by_tok, context_dims),
            self.SOURCE: self._calc_source_confidence(root, dim_by_tok, ent_label_by_tok),
            self.GEOGRAPHICAL: 2.5 if self.GEOGRAPHICAL in dim_by_tok.get(root.i, []) else 0.0
        }

    def _calc_subject_confidence(self, root: Token, dim_by_tok: dict[int, set[str]], context_dims: set[str]) -> float:
        """Calculate confidence for subject dimension (movie, series etc)."""
        confidence = 0.0

        # Direct dimension match is the strongest signal
        if self.SUBJECT in dim_by_tok.get(root.i, []):
            confidence += 2.5

        # POS tag analysis
        if root.pos_ == "NOUN":
            confidence += 1.0
            # Plural bonus (movies, films, shows are often plural)
            if root.tag_ in ("NNS", "NNPS"):
                confidence += 0.5

        # Dependency patterns
        confidence += self.dep_confidences["SUBJECT"].get(root.dep_, 0)

        # Context bonuses
        if self.CATEGORY in context_dims:  # "thriller movies"
            confidence += 0.6
        if self.GEOGRAPHICAL in context_dims:  # "Korean films"
            confidence += 0.4

        return confidence

    def _calc_category_confidence(self, root: Token, dim_by_tok: dict[int, set[str]], context_dims: set[str]) -> float:
        """Calculate confidence for category dimension."""
        confidence = 0.0
        tok_dimensions = dim_by_tok.get(root.i, [])

        # Skip if token is already of another dimension.
        if {self.SUBJECT, self.SOURCE, self.GEOGRAPHICAL} & tok_dimensions:
            return 0.0

        # Direct dimension match
        if self.CATEGORY in tok_dimensions:
            confidence += 2.5

        # POS patterns for genres
        if root.pos_ == "ADJ":  # Many genres are adjectives
            confidence += 1.0
            if root.dep_ == "amod":  # Adjectival modifier
                confidence += 0.5
        elif root.pos_ == "NOUN" and root.dep_ in ("compound", "nmod"):
            confidence += 1.5

        # Dependency patterns
        confidence += self.dep_confidences["CATEGORY"].get(root.dep_, 0)

        # Context analysis
        if self.SUBJECT in context_dims:  # Genres often modify subjects
            confidence += 0.7

        return confidence

    def _calc_source_confidence(
            self, root: Token,
            dim_by_tok: dict[int, set[str]],
            ent_label_by_tok: dict[int, set[str]]
    ) -> float:
        """Calculate confidence for source/platform dimension."""
        confidence = 0.0
        tok_dimensions = dim_by_tok.get(root.i, [])

        # Direct dimension match
        if self.SOURCE in tok_dimensions:
            confidence += 2.5

        # Proper nouns are often platforms
        if root.pos_ == "PROPN":
            confidence += 1.2

        # Organization entities
        if "ORG" in ent_label_by_tok.get(root.i, []):
            confidence += 1.0

        # Dependency patterns
        confidence += self.dep_confidences["SOURCE"].get(root.dep_, 0)

        return confidence

    def _select_subject(
            self, subj_label: str,
            data: dict[str, dict[str, float]],
            confidence_dict: dict[str, float]
    ) -> str:
        """Select the best subject with the highest confidence."""
        candidates = data[subj_label]
        default_subject = self.default_subject

        if not candidates:
            return default_subject

        # Find best candidate
        best_subject = max(candidates, key=candidates.get)
        best_confidence = candidates[best_subject]

        # Check threshold
        if best_confidence >= self.thresholds["SUBJECT"]:
            confidence_dict[subj_label] = best_confidence

            # Verify it's actually a subject dimension term
            if best_subject in self.nlp.vocab.strings:
                orth_id = self.nlp.vocab.strings[best_subject]
                if self.dim_by_canon_orth.get(orth_id) == subj_label:
                    return best_subject

        # Use default if no valid subject found
        return default_subject

    def _filter_dimension(
            self, dimension_label: str,
            data: dict[str, dict[str, float]],
            confidence_dict: dict[str, float]
    ) -> list[str]:
        """Filter and select terms for a dimension based on confidence threshold."""
        candidates = data[dimension_label]

        if not candidates:
            return []

        threshold = self.thresholds.get(
            self.dimension_names.get(dimension_label, "CATEGORY"),
            0.0
        )

        # Filter by threshold
        selected = []
        confidences = []

        for canonical, conf in candidates.items():
            if conf >= threshold:
                selected.append(canonical)
                confidences.append(conf)

        # Calculate dimension confidence
        if confidences:
            # Weighted combination of max and average
            max_conf = max(confidences)
            avg_conf = sum(confidences) / len(confidences)

            # 70% max, 30% average
            dimension_confidence = (0.7 * max_conf) + (0.3 * avg_conf)

            # Multi-match bonus
            if len(confidences) > 1:
                dimension_confidence += min(0.5, (len(confidences) - 1) * 0.25)

            confidence_dict[dimension_label] = dimension_confidence

        return selected

    def _calculate_overall_confidence(self, confidences: dict[str, float], doc: Doc) -> float:
        """
        Calculate overall domain confidence (0-100 scale).

        This represents how confident we are that the query belongs to this domain.
        """
        if not confidences:
            return 0.0

        # Map dimension labels to weights
        weights_map = {
            self.SUBJECT: self.dimension_weights["SUBJECT"],
            self.CATEGORY: self.dimension_weights["CATEGORY"],
            self.SOURCE: self.dimension_weights["SOURCE"],
            self.GEOGRAPHICAL: self.dimension_weights["GEOGRAPHICAL"],
        }

        # Calculate weighted score
        weighted_sum = 0.0
        total_weight = 0.0
        dimension_count = 0

        for label, weight in weights_map.items():
            if label in confidences and confidences[label] > 0:
                # Normalize confidence (assuming max ~6.0)
                normalized = min(confidences[label] / 6.0, 1.0)
                weighted_sum += normalized * weight
                total_weight += weight
                dimension_count += 1

        if total_weight == 0:
            return 0.0

        # Base confidence
        base_confidence = (weighted_sum / total_weight) * 100

        # Apply modifiers

        # Strong subject bonus
        if self.SUBJECT in confidences and confidences[self.SUBJECT] >= self.thresholds["SUBJECT"]:
            base_confidence *= 1.15  # 15% bonus for strong subject
        elif self.SUBJECT not in confidences or confidences[self.SUBJECT] == 0:
            base_confidence *= 0.6  # 40% penalty for missing subject

        # Multi-dimension bonus
        if dimension_count >= 3:
            base_confidence += 10  # Comprehensive query bonus
        elif dimension_count >= 2:
            base_confidence += 5  # Good query bonus
        elif dimension_count == 1 and self.SUBJECT not in confidences:
            base_confidence *= 0.7  # Single non-subject dimension penalty

        # Query length consideration
        query_length = len(doc)
        if query_length >= 5:  # Longer, more specific queries
            base_confidence += min(5, query_length - 4)

        # Cap at 100
        return min(100.0, max(0.0, base_confidence))