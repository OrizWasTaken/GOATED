import json
from pathlib import Path

from spacy.language import Language
from spacy.matcher import PhraseMatcher

def load_json(file_path: str):
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"\"{path}\" does not exist.")
    try:
        return json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"\"{path}\" is not a valid JSON file. {e}")


class DomainVocabManager:
    """Loads a domain-specific vocabulary from JSON and manages it for query parsing."""

    def __init__(self, file_path: str, nlp: Language):
        self.nlp = nlp
        self.config = load_json(file_path)
        self.vocabulary = self.config["vocabulary"]
        self.matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")

    def build_matcher(self):
        """Register all aliases into spaCy PhraseMatcher."""
        for entity_group in self.vocabulary.values():
            for canonical, aliases in entity_group.items():
                self.matcher.add(canonical, list(self.nlp.pipe(aliases)))
        return self.matcher