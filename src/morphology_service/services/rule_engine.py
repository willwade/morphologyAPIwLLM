from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from morphology_service.api.v1 import schemas
from morphology_service.services.hfst_backend import EnglishHFSTMorphology


@dataclass
class RuleEngine:
    """Finite-state English morphology orchestrated via HFST."""

    english: EnglishHFSTMorphology = field(default_factory=EnglishHFSTMorphology)

    def conjugate(
        self,
        lang: str,
        form: str,
        *,
        lemma: Optional[str] = None,
        expand_compound: bool = True,
        variety: Optional[str] = None,
    ) -> List[schemas.ConjugationParadigm]:
        if lang != "en":
            raise NotImplementedError("Rule-based conjugation currently supports English only.")
        return self.english.conjugate(
            form=form,
            lemma=lemma,
            expand_compound=expand_compound,
            variety=variety,
        )

    def lemmatize(self, lang: str, form: str, *, variety: Optional[str] = None) -> schemas.LemmaResponse:
        if lang != "en":
            raise NotImplementedError("Rule-based lemmatization currently supports English only.")

        return self.english.lemmatize(form=form, variety=variety)

    # --- Noun / adjective helpers -------------------------------------------------

    def inflect_number(self, lang: str, form: str, target_number: str) -> schemas.InflectionResult:
        if lang != "en":
            raise NotImplementedError("Rule-based inflection currently supports English only.")

        return self.english.inflect_number(form=form, target_number=target_number)

    # --- Auxiliary endpoints ------------------------------------------------------

    def number_to_words(self, lang: str, digits: str) -> schemas.NumberSpelling:
        if lang != "en":
            raise NotImplementedError("Rule-based number spelling currently supports English only.")

        return self.english.number_to_words(digits)

    def definitions(
        self,
        src_lang: str,
        tgt_lang: str,
        word: str,
    ) -> schemas.DefinitionResponse:
        entry = schemas.DefinitionItem(
            surfaceform=word,
            text=word,
            root=self.english._lemmatize_en(word) if src_lang == "en" else word,
            definition=f"Placeholder definition for '{word}'.",
            partofspeech=schemas.PartOfSpeechFeatures(partofspeechcategory="noun"),
            metadata=self.english._item_metadata(),
        )
        return schemas.DefinitionResponse(
            input=word,
            results=[entry],
            metadata=schemas.Metadata(provenance=schemas.Provenance.RULE, confidence=0.5),
        )

    # --- Catalog endpoints --------------------------------------------------------

    def list_languages(self) -> List[dict]:
        return self.english.list_languages()

    def list_dictionaries(self) -> List[dict]:
        return self.english.list_dictionaries()

    def list_parts_of_speech(self) -> List[str]:
        return self.english.list_parts_of_speech()

    def list_persons(self) -> List[str]:
        return self.english.list_persons()

    def list_tenses(self) -> List[str]:
        return self.english.list_tenses()
