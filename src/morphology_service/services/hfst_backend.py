from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from morphology_service.api.v1 import schemas

try:
    import hfst as _hfst  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _hfst = None


MORPHOLOGY_CONFIDENCE = 1.0


def _quote(symbol: str) -> str:
    escaped = symbol.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


@dataclass
class EnglishHFSTMorphology:
    """Finite-state morphology helper for English. Falls back to heuristics if HFST is unavailable."""

    irregular_plurals: Dict[str, List[str]] = field(default_factory=dict)
    irregular_verbs: Dict[str, Dict[str, str]] = field(default_factory=dict)
    _backend_note: str = field(init=False, repr=False)
    _hfst: Optional[object] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._hfst = _hfst
        self._backend_note = "hfst" if self._hfst else "deterministic-rules"
        if not self.irregular_plurals:
            self.irregular_plurals = {
                "mouse": ["mice"],
                "goose": ["geese"],
                "child": ["children"],
                "man": ["men"],
                "woman": ["women"],
                "tooth": ["teeth"],
                "foot": ["feet"],
                "person": ["people"],
            }
        if not self.irregular_verbs:
            self.irregular_verbs = {
                "be": {
                    "past": "was",
                    "past_plural": "were",
                    "past_participle": "been",
                    "present_participle": "being",
                    "third_person_singular": "is",
                },
                "have": {
                    "past": "had",
                    "past_participle": "had",
                    "present_participle": "having",
                    "third_person_singular": "has",
                },
                "do": {
                    "past": "did",
                    "past_participle": "done",
                    "present_participle": "doing",
                    "third_person_singular": "does",
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
                "run": {
                    "past": "ran",
                    "past_participle": "run",
                    "present_participle": "running",
                    "third_person_singular": "runs",
                },
                "write": {
                    "past": "wrote",
                    "past_participle": "written",
                    "present_participle": "writing",
                    "third_person_singular": "writes",
                },
            }

    # ------------------------------------------------------------------ Verbs --

    def conjugate(
        self,
        form: str,
        *,
        lemma: Optional[str] = None,
        expand_compound: bool,
        variety: Optional[str],
    ) -> List[schemas.ConjugationParadigm]:
        base = (lemma or self._lemmatize_en(form)).lower()
        tags_to_forms = self._build_verb_forms(base)
        transducer = self._compile_union(tags_to_forms.items()) if self._hfst else None
        paradigm_items: List[schemas.MorphologyItem] = []

        order = [
            ("+V+INF", {"tense": "infinitive"}),
            ("+V+PRS+P3+SG", {"tense": "present", "person": "third", "number": "singular"}),
            ("+V+PAST", {"tense": "past"}),
            ("+V+PAST+PTCP", {"tense": "pastparticiple"}),
            ("+V+PRS+PTCP", {"tense": "presentparticiple"}),
        ]
        if expand_compound:
            order.extend(
                [
                    ("+V+FUT", {"tense": "future", "mood": "indicative"}),
                    ("+V+PRS+PERF", {"tense": "presentperfect"}),
                ]
            )

        for tag_suffix, feature_overrides in order:
            key = f"{base}{tag_suffix}"
            surface_candidates = self._lookup(transducer, key, tags_to_forms.get(key))
            if not surface_candidates:
                continue
            surface = surface_candidates[0]
            paradigm_items.append(
                schemas.MorphologyItem(
                    surfaceform=surface,
                    text=surface,
                    root=base,
                    partofspeech=schemas.PartOfSpeechFeatures(partofspeechcategory="verb", **feature_overrides),
                    metadata=self._item_metadata(),
                )
            )

        metadata = schemas.Metadata(
            provenance=schemas.Provenance.RULE,
            confidence=MORPHOLOGY_CONFIDENCE,
            variety=variety,
            lemma_detected_from=form if base != form else None,
            notes=[f"generated via {self._backend_note}"],
        )

        return [
            schemas.ConjugationParadigm(
                infinitive=tags_to_forms.get(f"{base}+V+INF", base),
                conjugations=paradigm_items,
                metadata=metadata,
            )
        ]

    def lemmatize(
        self,
        form: str,
        *,
        variety: Optional[str] = None,
    ) -> schemas.LemmaResponse:
        lemma_guess = self._lemmatize_en(form)
        verb_forms = self._build_verb_forms(lemma_guess)
        lemma_candidates: List[schemas.LemmaCandidate] = []

        if self._hfst:
            transducer = self._compile_union(verb_forms.items())
            analyzer = transducer.copy()
            analyzer.invert()
            analyses = self._lookup(analyzer, form.lower(), None)
            for analysis in analyses:
                lemma = analysis.split("+", 1)[0]
                lemma_candidates.append(
                    schemas.LemmaCandidate(
                        lemma=lemma,
                        partofspeechcategory="verb",
                        confidence=MORPHOLOGY_CONFIDENCE,
                        metadata=self._item_metadata(),
                    )
                )
        else:
            for tag, surface in verb_forms.items():
                if surface == form.lower():
                    lemma = tag.split("+", 1)[0]
                    lemma_candidates.append(
                        schemas.LemmaCandidate(
                            lemma=lemma,
                            partofspeechcategory="verb",
                            confidence=MORPHOLOGY_CONFIDENCE,
                            metadata=self._item_metadata(),
                        )
                    )

        if not lemma_candidates:
            lemma_candidates.append(
                schemas.LemmaCandidate(
                    lemma=lemma_guess,
                    partofspeechcategory="verb",
                    confidence=0.9,
                    metadata=self._item_metadata(),
                )
            )

        response_metadata = schemas.Metadata(
            provenance=schemas.Provenance.RULE,
            confidence=max(candidate.confidence or 0 for candidate in lemma_candidates),
            variety=variety,
            notes=[f"generated via {self._backend_note}"],
        )
        return schemas.LemmaResponse(input=form, candidates=lemma_candidates, metadata=response_metadata)

    # -------------------------------------------------------------- Nouns/Adjs --

    def inflect_number(self, form: str, target_number: str) -> schemas.InflectionResult:
        lemma = self._lemmatize_noun(form if target_number == "plural" else self._to_singular(form))
        noun_forms = self._build_noun_forms(lemma)
        transducer = self._compile_union(noun_forms.items()) if self._hfst else None
        key_suffix = "+N+PL" if target_number == "plural" else "+N+SG"
        outputs = self._lookup(transducer, f"{lemma}{key_suffix}", noun_forms.get(f"{lemma}{key_suffix}"))
        if not outputs:
            outputs = [form]

        features = schemas.PartOfSpeechFeatures(partofspeechcategory="noun", number=target_number)
        metadata = schemas.Metadata(
            provenance=schemas.Provenance.RULE,
            confidence=MORPHOLOGY_CONFIDENCE,
            notes=[f"generated via {self._backend_note}"],
        )
        return schemas.InflectionResult(input=form, results=outputs, features=features, metadata=metadata)

    # ---------------------------------------------------------------- Utilities --

    def number_to_words(self, digits: str) -> schemas.NumberSpelling:
        text = self._number_to_words_en(digits)
        transducer = self._compile_union([(digits, text)]) if self._hfst else None
        outputs = self._lookup(transducer, digits, text)
        surface = outputs[0] if outputs else text
        return schemas.NumberSpelling(
            input=digits,
            text=surface,
            metadata=schemas.Metadata(
                provenance=schemas.Provenance.RULE,
                confidence=MORPHOLOGY_CONFIDENCE,
                notes=[f"generated via {self._backend_note}"],
            ),
        )

    # --------------------------------------------------------------- Catalogues --

    def list_languages(self) -> List[dict]:
        backend = self._backend_note
        return [
            {"english_name": "English", "iso2": "en", "iso3": "eng", "backend": backend},
            {"english_name": "Spanish", "iso2": "es", "iso3": "spa"},
        ]

    def list_dictionaries(self) -> List[dict]:
        backend = self._backend_note
        return [
            {"id": "en-en", "source": "english", "target": "english", "variant": "general", "backend": backend},
            {"id": "en-es", "source": "english", "target": "spanish"},
        ]

    @staticmethod
    def list_parts_of_speech() -> List[str]:
        return ["noun", "verb", "adjective", "adverb", "pronoun", "preposition"]

    @staticmethod
    def list_persons() -> List[str]:
        return ["first", "second", "third"]

    @staticmethod
    def list_tenses() -> List[str]:
        return [
            "infinitive",
            "present",
            "past",
            "future",
            "presentperfect",
            "pastparticiple",
            "presentparticiple",
        ]

    # ---------------------------------------------------------- Helper methods --

    def _compile_union(self, pairs: Iterable[Tuple[str, str]]):
        assert self._hfst, "HFST must be available to compile transducers."
        transducer = None
        for left, right in pairs:
            pair_transducer = self._hfst.regex(f"{_quote(left)}:{_quote(right)}")
            transducer = pair_transducer if transducer is None else transducer.disjunct(pair_transducer)
        if transducer is None:
            transducer = self._hfst.epsilon_fst()
        transducer.convert(self._hfst.ImplementationType.HFST_OL_TYPE)
        return transducer

    def _lookup(self, transducer, key: str, default: Optional[str]) -> List[str]:
        if self._hfst and transducer is not None:
            results = transducer.lookup(key, output="tuple")
            if results:
                return [surface for surface, _ in results]
        return [default] if default else []

    def _build_verb_forms(self, lemma: str) -> Dict[str, str]:
        lemma = lemma.lower()
        irregular = self.irregular_verbs.get(lemma, {})

        third_person = irregular.get("third_person_singular") or self._append_third_person(lemma)
        past = irregular.get("past") or self._append_past(lemma)
        past_participle = irregular.get("past_participle") or past
        gerund = irregular.get("present_participle") or self._append_gerund(lemma)

        forms = {
            f"{lemma}+V+INF": lemma,
            f"{lemma}+V+PRS+P3+SG": third_person,
            f"{lemma}+V+PAST": past,
            f"{lemma}+V+PAST+PTCP": past_participle,
            f"{lemma}+V+PRS+PTCP": gerund,
            f"{lemma}+V+FUT": f"will {lemma}",
            f"{lemma}+V+PRS+PERF": f"have {past_participle}",
        }

        if lemma == "be":
            forms[f"{lemma}+V+PAST"] = "was"
            forms[f"{lemma}+V+PAST+PL"] = "were"
            forms[f"{lemma}+V+PAST+PTCP"] = "been"
            forms[f"{lemma}+V+PRS+P1+SG"] = "am"
            forms[f"{lemma}+V+PRS+P2"] = "are"
            forms[f"{lemma}+V+PRS+P3+SG"] = "is"
            forms[f"{lemma}+V+PRS+P3+PL"] = "are"

        return forms

    def _build_noun_forms(self, lemma: str) -> Dict[str, str]:
        lemma = lemma.lower()
        plural_forms = self.irregular_plurals.get(lemma)
        plural = plural_forms[0] if plural_forms else self._pluralize_en(lemma)
        return {
            f"{lemma}+N+SG": lemma,
            f"{lemma}+N+PL": plural,
        }

    def _item_metadata(self) -> schemas.Metadata:
        return schemas.Metadata(provenance=schemas.Provenance.RULE, confidence=MORPHOLOGY_CONFIDENCE, notes=[self._backend_note])

    # --------------------------------------------------------------- Inflection --

    def _lemmatize_en(self, form: str) -> str:
        lowered = form.lower()
        for lemma, forms in self.irregular_verbs.items():
            if lowered == lemma or lowered in forms.values():
                return lemma
        if lowered.endswith("ing") and len(lowered) > 4:
            candidate = lowered[:-3]
            if len(candidate) > 2 and candidate[-1] == candidate[-2] and candidate[-1] not in "aeiou":
                candidate = candidate[:-1]
            return candidate
        if lowered.endswith("ied"):
            return lowered[:-3] + "y"
        if lowered.endswith("ed") and len(lowered) > 3:
            return lowered[:-2]
        if lowered.endswith("s") and not lowered.endswith("ss"):
            return lowered[:-1]
        return lowered

    def _lemmatize_noun(self, form: str) -> str:
        lowercase = form.lower()
        for singular, plurals in self.irregular_plurals.items():
            if lowercase == singular or lowercase in plurals:
                return singular
        if lowercase.endswith("ies"):
            return lowercase[:-3] + "y"
        if lowercase.endswith("es") and not lowercase.endswith("ses"):
            return lowercase[:-2]
        if lowercase.endswith("s") and not lowercase.endswith("ss"):
            return lowercase[:-1]
        return lowercase

    def _to_singular(self, plural_form: str) -> str:
        lowercase = plural_form.lower()
        for singular, plurals in self.irregular_plurals.items():
            if lowercase in plurals:
                return singular
        if lowercase.endswith("ies"):
            return lowercase[:-3] + "y"
        if lowercase.endswith("es"):
            return lowercase[:-2]
        if lowercase.endswith("s"):
            return lowercase[:-1]
        return lowercase

    def _append_third_person(self, lemma: str) -> str:
        if lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
            return lemma[:-1] + "ies"
        if lemma.endswith(("s", "sh", "ch", "x", "z", "o")):
            return lemma + "es"
        return lemma + "s"

    def _append_past(self, lemma: str) -> str:
        if lemma.endswith("e"):
            return lemma + "d"
        if lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
            return lemma[:-1] + "ied"
        if len(lemma) >= 3 and lemma[-1] not in "aeiou" and lemma[-2] in "aeiou" and lemma[-3] not in "aeiou":
            return lemma + lemma[-1] + "ed"
        return lemma + "ed"

    def _append_gerund(self, lemma: str) -> str:
        if lemma.endswith("ie"):
            return lemma[:-2] + "ying"
        if lemma.endswith("e") and lemma != "be":
            return lemma[:-1] + "ing"
        if len(lemma) >= 3 and lemma[-1] not in "aeiou" and lemma[-2] in "aeiou" and lemma[-3] not in "aeiou":
            return lemma + lemma[-1] + "ing"
        return lemma + "ing"

    def _pluralize_en(self, word: str) -> str:
        if word in self.irregular_plurals:
            return self.irregular_plurals[word][0]
        if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
            return word[:-1] + "ies"
        if word.endswith(("s", "sh", "ch", "x", "z", "o")):
            return word + "es"
        return word + "s"

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
        units = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
        tens_words = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

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
