from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from morphology_service.api.v1 import schemas


@dataclass
class RuleEngine:
    """Lightweight deterministic heuristics for English morphology."""

    def __post_init__(self) -> None:
        self.irregular_plurals: Dict[str, List[str]] = {
            "mouse": ["mice"],
            "goose": ["geese"],
            "child": ["children"],
            "man": ["men"],
            "woman": ["women"],
            "tooth": ["teeth"],
            "foot": ["feet"],
        }
        self.irregular_verbs: Dict[str, Dict[str, str]] = {
            "be": {
                "past": "was",
                "past_plural": "were",
                "past_participle": "been",
                "present_participle": "being",
                "third_person_singular": "is",
            },
            "go": {
                "past": "went",
                "past_participle": "gone",
                "present_participle": "going",
                "third_person_singular": "goes",
            },
            "sleep": {
                "past": "slept",
                "past_participle": "slept",
                "present_participle": "sleeping",
                "third_person_singular": "sleeps",
            },
        }

    # --- Verb conjugation helpers -------------------------------------------------

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

        lemma = lemma or self._lemmatize_en(form)
        irregular = self.irregular_verbs.get(lemma, {})

        base = lemma
        third_person = irregular.get("third_person_singular") or self._append_third_person(base)
        past = irregular.get("past") or self._append_past(base)
        past_participle = irregular.get("past_participle") or past
        gerund = irregular.get("present_participle") or self._append_gerund(base)

        paradigm = schemas.ConjugationParadigm(
            infinitive=base,
            conjugations=[
                schemas.MorphologyItem(
                    surfaceform=base,
                    text=base,
                    root=base,
                    partofspeech=schemas.PartOfSpeechFeatures(partofspeechcategory="verb", tense="infinitive"),
                ),
                schemas.MorphologyItem(
                    surfaceform=third_person,
                    text=third_person,
                    root=base,
                    partofspeech=schemas.PartOfSpeechFeatures(
                        partofspeechcategory="verb",
                        tense="present",
                        person="third",
                        number="singular",
                    ),
                ),
                schemas.MorphologyItem(
                    surfaceform=past,
                    text=past,
                    root=base,
                    partofspeech=schemas.PartOfSpeechFeatures(
                        partofspeechcategory="verb",
                        tense="past",
                    ),
                ),
                schemas.MorphologyItem(
                    surfaceform=past_participle,
                    text=past_participle,
                    root=base,
                    partofspeech=schemas.PartOfSpeechFeatures(
                        partofspeechcategory="verb",
                        tense="pastparticiple",
                    ),
                ),
                schemas.MorphologyItem(
                    surfaceform=gerund,
                    text=gerund,
                    root=base,
                    partofspeech=schemas.PartOfSpeechFeatures(
                        partofspeechcategory="verb",
                        tense="presentparticiple",
                    ),
                ),
            ],
            metadata=schemas.Metadata(
                provenance=schemas.Provenance.RULE,
                confidence=1.0,
                variety=variety,
                lemma_detected_from=form if lemma != form else None,
            ),
        )

        if expand_compound:
            paradigm.conjugations.append(
                schemas.MorphologyItem(
                    surfaceform=f"will {base}",
                    text=f"will {base}",
                    root=base,
                    partofspeech=schemas.PartOfSpeechFeatures(
                        partofspeechcategory="verb",
                        tense="future",
                        mood="indicative",
                    ),
                )
            )
            paradigm.conjugations.append(
                schemas.MorphologyItem(
                    surfaceform=f"have {past_participle}",
                    text=f"have {past_participle}",
                    root=base,
                    partofspeech=schemas.PartOfSpeechFeatures(
                        partofspeechcategory="verb",
                        tense="presentperfect",
                    ),
                )
            )

        return [paradigm]

    def lemmatize(self, lang: str, form: str, *, variety: Optional[str] = None) -> schemas.LemmaResponse:
        if lang != "en":
            raise NotImplementedError("Rule-based lemmatization currently supports English only.")

        lemma = self._lemmatize_en(form)
        candidate = schemas.LemmaCandidate(
            lemma=lemma,
            partofspeechcategory="verb",
            confidence=1.0 if lemma == form or lemma in self.irregular_verbs else 0.9,
        )
        return schemas.LemmaResponse(
            input=form,
            candidates=[candidate],
            metadata=schemas.Metadata(
                provenance=schemas.Provenance.RULE,
                confidence=candidate.confidence,
                variety=variety,
            ),
        )

    # --- Noun / adjective helpers -------------------------------------------------

    def inflect_number(self, lang: str, form: str, target_number: str) -> schemas.InflectionResult:
        if lang != "en":
            raise NotImplementedError("Rule-based inflection currently supports English only.")

        if target_number == "plural":
            results = self._pluralize_en(form)
            features = schemas.PartOfSpeechFeatures(partofspeechcategory="noun", number="plural")
        else:
            results = self._singularize_en(form)
            features = schemas.PartOfSpeechFeatures(partofspeechcategory="noun", number="singular")

        return schemas.InflectionResult(
            input=form,
            results=results,
            features=features,
            metadata=schemas.Metadata(
                provenance=schemas.Provenance.RULE,
                confidence=1.0,
            ),
        )

    # --- Auxiliary endpoints ------------------------------------------------------

    def number_to_words(self, lang: str, digits: str) -> schemas.NumberSpelling:
        if lang != "en":
            raise NotImplementedError("Rule-based number spelling currently supports English only.")

        text = self._number_to_words_en(digits)
        return schemas.NumberSpelling(
            input=digits,
            text=text,
            metadata=schemas.Metadata(provenance=schemas.Provenance.RULE, confidence=1.0),
        )

    def definitions(
        self,
        src_lang: str,
        tgt_lang: str,
        word: str,
    ) -> schemas.DefinitionResponse:
        entry = schemas.DefinitionItem(
            surfaceform=word,
            text=word,
            root=self._lemmatize_en(word) if src_lang == "en" else word,
            definition=f"Placeholder definition for '{word}'.",
            partofspeech=schemas.PartOfSpeechFeatures(partofspeechcategory="noun"),
        )
        return schemas.DefinitionResponse(
            input=word,
            results=[entry],
            metadata=schemas.Metadata(provenance=schemas.Provenance.RULE, confidence=0.5),
        )

    # --- Catalog endpoints --------------------------------------------------------

    def list_languages(self) -> List[dict]:
        return [
            {"english_name": "English", "iso2": "en", "iso3": "eng"},
            {"english_name": "Spanish", "iso2": "es", "iso3": "spa"},
        ]

    def list_dictionaries(self) -> List[dict]:
        return [
            {"id": "en-en", "source": "english", "target": "english", "variant": "general"},
            {"id": "en-es", "source": "english", "target": "spanish"},
        ]

    def list_parts_of_speech(self) -> List[str]:
        return ["noun", "verb", "adjective", "adverb", "pronoun", "preposition"]

    def list_persons(self) -> List[str]:
        return ["first", "second", "third"]

    def list_tenses(self) -> List[str]:
        return [
            "infinitive",
            "present",
            "past",
            "future",
            "presentperfect",
            "pastperfect",
            "futureperfect",
        ]

    # --- Internal utilities -------------------------------------------------------

    def _lemmatize_en(self, form: str) -> str:
        lowered = form.lower()
        for lemma, forms in self.irregular_verbs.items():
            if lowered == lemma or lowered in forms.values():
                return lemma
        if lowered.endswith("ing") and len(lowered) > 4:
            return re.sub("ing$", "", lowered)
        if lowered.endswith("ied"):
            return lowered[:-3] + "y"
        if lowered.endswith("ed") and len(lowered) > 3:
            return lowered[:-2]
        if lowered.endswith("s") and not lowered.endswith("ss"):
            return lowered[:-1]
        return lowered

    def _append_third_person(self, lemma: str) -> str:
        if lemma.endswith("y") and lemma[-2] not in "aeiou":
            return lemma[:-1] + "ies"
        if lemma.endswith(("s", "sh", "ch", "x", "z", "o")):
            return lemma + "es"
        return lemma + "s"

    def _append_past(self, lemma: str) -> str:
        if lemma.endswith("e"):
            return lemma + "d"
        if lemma.endswith("y") and lemma[-2] not in "aeiou":
            return lemma[:-1] + "ied"
        return lemma + "ed"

    def _append_gerund(self, lemma: str) -> str:
        if lemma.endswith("e") and lemma != "be":
            return lemma[:-1] + "ing"
        return lemma + "ing"

    def _pluralize_en(self, word: str) -> List[str]:
        lowercase = word.lower()
        if lowercase in self.irregular_plurals:
            return self.irregular_plurals[lowercase]
        if lowercase.endswith("y") and lowercase[-2] not in "aeiou":
            return [lowercase[:-1] + "ies"]
        if lowercase.endswith(("s", "sh", "ch", "x", "z")):
            return [lowercase + "es"]
        return [lowercase + "s"]

    def _singularize_en(self, word: str) -> List[str]:
        lowercase = word.lower()
        for singular, plurals in self.irregular_plurals.items():
            if lowercase in plurals:
                return [singular]
        if lowercase.endswith("ies"):
            return [lowercase[:-3] + "y"]
        if lowercase.endswith("es"):
            return [lowercase[:-2]]
        if lowercase.endswith("s"):
            return [lowercase[:-1]]
        return [lowercase]

    def _number_to_words_en(self, digits: str) -> str:
        try:
            value = int(digits)
        except ValueError:
            return digits
        if value == 0:
            return "zero"
        parts: List[str] = []
        if value < 0:
            parts.append("negative")
            value = abs(value)
        units = [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        ]
        teens = [
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]
        tens_words = [
            "",
            "",
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "sixty",
            "seventy",
            "eighty",
            "ninety",
        ]

        def under_hundred(n: int) -> str:
            if n < 10:
                return units[n]
            if n < 20:
                return teens[n - 10]
            ten, unit = divmod(n, 10)
            return tens_words[ten] if unit == 0 else f"{tens_words[ten]}-{units[unit]}"

        def under_thousand(n: int) -> str:
            hundred, rem = divmod(n, 100)
            if hundred and rem:
                return f"{units[hundred]} hundred and {under_hundred(rem)}"
            if hundred:
                return f"{units[hundred]} hundred"
            return under_hundred(rem)

        billions, rem = divmod(value, 1_000_000_000)
        millions, rem = divmod(rem, 1_000_000)
        thousands, rem = divmod(rem, 1_000)

        if billions:
            parts.append(f"{under_thousand(billions)} billion")
        if millions:
            parts.append(f"{under_thousand(millions)} million")
        if thousands:
            parts.append(f"{under_thousand(thousands)} thousand")
        if rem:
            parts.append(under_thousand(rem))
        return ", ".join(parts)
