import json
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Set, Tuple

from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Token, Doc, Span


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

    # Dependency confidence maps optimized for movie domain
    dep_confidences = {
        "SUBJECT": {
            "ROOT": 1.5,  # Main verb object (e.g., "watch movies")
            "dobj": 1.2,  # Direct object (e.g., "best movies")
            "nsubj": 0.8, # Nominal subject
            "pobj": 0.2,  # Object of preposition (e.g., "for movies")
            "attr": 0.15, # Attribute
        },
        "CATEGORY": {
            "conj": 0.5,  # Conjunction (e.g., "action and thriller")
            "appos": 0.4  # Apposition
        },
        "SOURCE": {
            "pobj": 1.0,  # Object of preposition (e.g., "on Netflix")
            "prep": 0.8,  # Prepositional modifier
            "nmod": 0.6,  # Nominal modifier
            "conj": 0.4   # Conjunction (e.g., "Netflix and Hulu")
        },
        "TIMEFRAME": {
            "nummod": 1.0,  # Numeric modifier (e.g., "2024 movies")
            "amod": 0.8,    # Adjectival modifier (e.g., "recent movies")
            "pobj": 0.6,    # Object of preposition (e.g., "from 2020s")
            "nmod": 0.5,    # Nominal modifier
            "npadvmod": 0.7 # Noun phrase as adverbial modifier
        },
        "GEOGRAPHICAL": {
            "compound": 0.8,  # Compound (e.g., "Bollywood films")
            "nmod": 0.6,      # Nominal modifier
            "pobj": 0.5       # Object of preposition
        }
    }

    # Confidence thresholds for dimension selection
    thresholds = {
        "SUBJECT": 2.0,      # Min: pass tag analysis + reasonable dependency pattern.
        "CATEGORY": 2.0,     # Min: pass POS patterns + context analysis.
        "SOURCE": 2.5,       # High emphasis on domain's vocabulary.
        "GEOGRAPHICAL": 1.0,
        "TIMEFRAME": 0.8,
    }

    # Weights for overall confidence calculation
    dimension_weights = {
        "SUBJECT": 0.50,  # Type (movie/series) is very important (50%)
        "CATEGORY": 0.25,  # Genre is quite important (25%)
        "SOURCE": 0.15,  # Platform is moderately important (15%)
        "GEOGRAPHICAL": 0.10,  # Region is less important (10%)
        "TIMEFRAME": 0.0,  # Era isn't important (0%)
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
        labels = self.config["dimension_labels"]
        self.SUBJECT = labels["SUBJECT"] # Required dimension, raise KeyError if missing.
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
    def dim_by_canon_orth(self) -> Dict[int, str]:
        """Map each canonical term's orth ID to its dimension."""
        _ = self.matcher  # Force matcher initialization

        mapping = {}
        for dimension, canonicals in self.vocabulary.items():
            for canonical in canonicals:
                orth_id = self.nlp.vocab.strings.add(canonical)
                mapping[orth_id] = dimension

        return mapping

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
        confidences_dict = {}

        # Pre-compute mappings for efficiency
        dim_by_tok = self._dim_by_tok(matches)
        ent_label_by_tok = self._ent_label_by_tok(doc)

        # Process each match
        for match_id, start, end in matches:
            span = doc[start:end]

            # Skip invalid POS tags
            if span.root.pos_ not in self.VALID_POS_TAGS:
                continue

            dimension = self.dim_by_canon_orth.get(match_id)
            if not dimension:
                continue

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

        # Format final results
        result = {
            self.SUBJECT: self._select_subject(data, confidences_dict),
            self.CATEGORY: self._filter_dimension(self.CATEGORY, data, confidences_dict),
            self.SOURCE: self._filter_dimension(self.SOURCE, data, confidences_dict),
            self.GEOGRAPHICAL: self._filter_dimension(self.GEOGRAPHICAL, data, confidences_dict),
            self.TIMEFRAME: self._filter_dimension(self.TIMEFRAME, data, confidences_dict),
            "confidence": self._calculate_overall_confidence(confidences_dict, doc)
        }

        return result

    def _dim_by_tok(self, matches: List[Tuple[int, int, int]]) -> Dict[int, Set[str]]:
        """Map token indices to their dimensions."""
        dim_by_tok = defaultdict(set)

        for match_id, start, end in matches:
            dimension = self.dim_by_canon_orth.get(match_id)
            if dimension:
                for i in range(start, end):
                    dim_by_tok[i].add(dimension)

        return dict(dim_by_tok)

    @staticmethod
    def _ent_label_by_tok(doc: Doc) -> Dict[int, Set[str]]:
        """Map token indices to their spaCy entity labels."""
        ent_by_tok = defaultdict(set)

        for ent in doc.ents:
            for i in range(ent.start, ent.end):
                ent_by_tok[i].add(ent.label_)

        return dict(ent_by_tok)

    def _calculate_all_confidences(
            self, span: Span, doc: Doc,
            dim_by_tok: Dict[int, Set[str]],
            ent_label_by_tok: Dict[int, Set[str]]
    ) -> Dict[str, float]:
        """Calculate confidence scores for all dimensions."""
        root = span.root
        context_tokens = self._get_context_tokens(span, doc, window=1)

        # Analyze context
        context_dims = set()
        for token in context_tokens:
            if token.i != root.i:  # Exclude the root token itself
                context_dims.update(dim_by_tok.get(token.i, []))

        return {
            self.SUBJECT: self._calc_subject_confidence(root, dim_by_tok, context_dims),
            self.CATEGORY: self._calc_category_confidence(root, dim_by_tok, context_dims),
            self.SOURCE: self._calc_source_confidence(root, dim_by_tok, ent_label_by_tok),
            self.GEOGRAPHICAL: self._calc_geographical_confidence(root, span, dim_by_tok, ent_label_by_tok),
            self.TIMEFRAME: self._calc_timeframe_confidence(root, span, dim_by_tok),
        }

    def _calc_subject_confidence(self, root: Token, dim_by_tok: Dict[int, Set[str]], context_dims: Set[str]) -> float:
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

    def _calc_category_confidence(self, root: Token, dim_by_tok: Dict[int, Set[str]], context_dims: Set[str]) -> float:
        """Calculate confidence for category dimension."""
        confidence = 0.0
        tok_dimensions = dim_by_tok.get(root.i, [])

        # Skip if token is already of another dimension.
        if {self.SUBJECT, self.SOURCE, self.GEOGRAPHICAL, self.TIMEFRAME} & tok_dimensions:
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
            dim_by_tok: Dict[int, Set[str]],
            ent_label_by_tok: Dict[int, Set[str]]
    ) -> float:
        """Calculate confidence for source/platform dimension."""
        confidence = 0.0
        tok_dimensions = dim_by_tok.get(root.i, [])

        if {self.SUBJECT, self.CATEGORY, self.GEOGRAPHICAL, self.TIMEFRAME} & tok_dimensions:
            return 0.0

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

    def _calc_geographical_confidence(
            self, root: Token,
            dim_by_tok: Dict[int, Set[str]],
            ent_label_by_tok: Dict[int, Set[str]]
    ) -> float:
        """Calculate confidence for geographical/region dimension."""
        confidence = 0.0
        tok_dimensions = dim_by_tok.get(root.i, [])

        # Skip if token is source or category
        if {self.SOURCE, self.CATEGORY} & tok_dimensions:
            return 0.0

        # Direct dimension match
        if self.GEOGRAPHICAL in tok_dimensions:
            confidence += 2.0

        # Entity type analysis
        tok_labels = ent_label_by_tok.get(root.i, [])
        if "GPE" in tok_labels:  # Geopolitical entity
            confidence += 1.5
        if "NORP" in tok_labels:  # Nationality
            confidence += 1.0

        # Adjectives that are often geographical
        if root.pos_ == "ADJ" and root.dep_ == "amod":
            # Check if it modifies a subject term
            if root.head and self.SUBJECT in dim_by_tok.get(root.head.i, []):
                confidence += 0.8

        # Dependency patterns
        confidence += self.dep_confidences["GEOGRAPHICAL"].get(root.dep_, 0)

        return confidence

    def _calc_timeframe_confidence(
            self, root: Token, span: Span,
            dim_by_tok: Dict[int, Set[str]]
    ) -> float:
        """Calculate confidence for timeframe/era dimension."""
        confidence = 0.0

        # Direct dimension match
        if self.TIMEFRAME in dim_by_tok.get(root.i, []):
            confidence += 2.5

            # Era-specific bonuses
            era_text = span.text.lower()
            if era_text in ["latest", "recent", "new", "classic"]:
                confidence += 0.5

        # Temporal adjectives
        if root.pos_ == "ADJ":
            temporal_adjs = {"recent", "latest", "new", "old", "classic", "modern",
                             "contemporary", "current", "upcoming", "retro", "vintage"}
            if root.text.lower() in temporal_adjs:
                confidence += 1.5

        # Year/decade patterns
        text = root.text
        if root.like_num or text.isdigit():
            # Check for year patterns (1990-2099)
            if len(text) == 4 and text.startswith(("19", "20")):
                confidence += 2.0
            # Decade patterns (90s, 2010s)
            elif text.endswith("s") and text[:-1].isdigit():
                confidence += 1.8

        # Date entities
        if root.ent_type_ == "DATE":
            confidence += 1.5

        # Dependency patterns
        confidence += self.dep_confidences["TIMEFRAME"].get(root.dep_, 0)

        return confidence

    def _select_subject(
            self, data: Dict[str, Dict[str, float]],
            confidence_dict: Dict[str, float]
    ) -> str:
        """Select the best subject with the highest confidence."""
        candidates = data[self.SUBJECT]
        default_subject = self.default_subject

        if not candidates:
            return default_subject

        # Find best candidate
        best_subject = max(candidates, key=candidates.get)
        best_confidence = candidates[best_subject]

        # Check threshold
        if best_confidence >= self.thresholds["SUBJECT"]:
            confidence_dict[self.SUBJECT] = best_confidence

            # Verify it's actually a subject dimension term
            if best_subject in self.nlp.vocab.strings:
                orth_id = self.nlp.vocab.strings[best_subject]
                if self.dim_by_canon_orth.get(orth_id) == self.SUBJECT:
                    return best_subject

        # Use default if no valid subject found
        return default_subject

    def _filter_dimension(
            self, dimension_label: str,
            data: Dict[str, Dict[str, float]],
            confidence_dict: Dict[str, float]
    ) -> List[str]:
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

    def _calculate_overall_confidence(
            self, confidences: Dict[str, float],
            doc: Doc
    ) -> float:
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
            self.TIMEFRAME: self.dimension_weights["TIMEFRAME"],
        }

        # Calculate weighted score
        weighted_sum = 0.0
        total_weight = 0.0
        dimension_count = 0

        for label, weight in weights_map.items():
            if label in confidences and confidences[label] > 0:
                # Normalize confidence (assuming max ~5.0)
                normalized = min(confidences[label] / 5.0, 1.0)
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

    @staticmethod
    def _get_context_tokens(span: Span, doc: Doc, window: int = 2) -> Span:
        """Get surrounding tokens within window size."""
        start = max(0, span.start - window)
        end = min(len(doc), span.end + window + 1)
        return doc[start:end]