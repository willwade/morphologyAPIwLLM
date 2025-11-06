from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import llm
from tenacity import retry, stop_after_attempt, wait_exponential

from morphology_service.api.v1 import schemas as api_schemas
from morphology_service.core.config import Settings


@dataclass
class LLMClient:
    """Wrapper around Simon Willison's `llm` library with JSON parsing helpers."""

    settings: Settings

    def __post_init__(self) -> None:
        if self.settings.gemini_api_key:
            os.environ.setdefault("GEMINI_API_KEY", self.settings.gemini_api_key)
        self.model = llm.get_model(self.settings.llm_model_name)

    def _build_prompt(
        self,
        instruction: str,
        payload: Dict[str, Any],
        *,
        example: Optional[Dict[str, Any]] = None,
    ) -> str:
        sections = [
            "SYSTEM: You are a morphology generator that returns valid JSON matching the requested schema.",
            f"INSTRUCTION: {instruction}",
            "REQUIREMENTS:",
            "- Respond with JSON only; no markdown or prose.",
            "- Use the field names and nesting shown in RESPONSE_FORMAT.",
            "- Populate metadata.provenance with \"llm\" unless instructed otherwise.",
            "- Provide confidence scores between 0 and 1 when possible.",
            "- When the payload supplies allowed_* lists, pick values exclusively from those lists.",
            "- Keep `text` identical to `surfaceform` unless the payload explicitly requests display variants.",
            f"PAYLOAD:\n{json.dumps(payload, ensure_ascii=False, indent=2)}",
            (
                f"RESPONSE_FORMAT:\n{json.dumps(example, ensure_ascii=False, indent=2)}"
                if example
                else "RESPONSE_FORMAT:\n{}"
            ),
            "Return ONLY JSON.",
        ]
        return "\n".join(sections)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
    def _generate_json(self, prompt: str) -> Dict[str, Any]:
        response = self.model.prompt(prompt, temperature=self.settings.llm_temperature)
        text = response.text()
        stripped = text.strip()
        if stripped.startswith("```"):
            # Remove Markdown code fences if the model included them.
            stripped = stripped.strip("`")
            if stripped.startswith("json"):
                stripped = stripped[4:]
            text = stripped.strip()
        else:
            text = stripped
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned invalid JSON: {text}") from exc

    @staticmethod
    def _coerce_list(container: Any) -> List[Dict[str, Any]]:
        if isinstance(container, list):
            return container
        if isinstance(container, dict):
            return [container]
        raise ValueError("Expected list or dict payload from LLM response.")

    @staticmethod
    def _ensure_list_field(data: Dict[str, Any], *, field: str, fallback_field: Optional[str] = None) -> None:
        if field in data and isinstance(data[field], list):
            return
        if field in data and isinstance(data[field], dict):
            data[field] = [data[field]]
            return
        if fallback_field and fallback_field in data:
            value = data.pop(fallback_field)
            data[field] = value if isinstance(value, list) else [value]
            return
        data[field] = []

    def _ensure_metadata(self, item: Dict[str, Any], default_provenance: str = "llm") -> None:
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            item["metadata"] = self._default_metadata(default_provenance)
            return
        metadata.setdefault("provenance", default_provenance)
        metadata.setdefault("confidence", None)
        metadata.setdefault("variety", None)
        metadata.setdefault("lemma_detected_from", None)
        metadata.setdefault("warnings", [])
        metadata.setdefault("notes", [])

    @staticmethod
    def _default_metadata(default_provenance: str) -> Dict[str, Any]:
        return {
            "provenance": default_provenance,
            "confidence": 0.9,
            "variety": None,
            "lemma_detected_from": None,
            "warnings": [],
            "notes": [],
        }

    async def generate_conjugations(
        self,
        lang: str,
        form: str,
        *,
        lemma: Optional[str],
        expand_compound: bool,
        variety: Optional[str],
        max_variants: int,
        return_features: Optional[List[str]],
        source_preference: str,
        force_deterministic: bool,
    ) -> List[api_schemas.ConjugationParadigm]:
        allowed_tenses = [
            "infinitive",
            "present",
            "past",
            "future",
            "presentperfect",
            "pastperfect",
            "futureperfect",
            "presentparticiple",
            "pastparticiple",
        ]
        allowed_persons = ["first", "second", "third"]
        allowed_numbers = ["singular", "plural"]
        example = {
            "paradigms": [
                {
                    "infinitive": "sleep",
                    "conjugations": [
                        {
                            "surfaceform": "sleep",
                            "text": "sleep",
                            "root": "sleep",
                            "partofspeech": {
                                "partofspeechcategory": "verb",
                                "tense": "present",
                                "person": "first",
                                "number": "singular",
                            },
                            "metadata": None,
                        }
                    ],
                    "metadata": {
                        "provenance": "llm",
                        "confidence": 0.95,
                        "variety": variety,
                        "lemma_detected_from": form,
                        "warnings": [],
                        "notes": [],
                    },
                }
            ]
        }
        prompt = self._build_prompt(
            "Generate a complete verb conjugation paradigm using the allowed feature values.",
            {
                "lang": lang,
                "form": form,
                "lemma": lemma,
                "expand_compound": expand_compound,
                "variety": variety,
                "max_variants": max_variants,
                "return_features": return_features,
                "source_preference": source_preference,
                "force_deterministic": force_deterministic,
                "allowed_tenses": allowed_tenses,
                "allowed_persons": allowed_persons,
                "allowed_numbers": allowed_numbers,
            },
            example=example,
        )
        data = await asyncio.to_thread(self._generate_json, prompt)
        if isinstance(data, list):
            paradigms_raw = data
        elif "paradigms" in data:
            paradigms_raw = data["paradigms"]
        else:
            paradigms_raw = [data]
        paradigms_list = self._coerce_list(paradigms_raw)
        for item in paradigms_list:
            self._ensure_metadata(item)
        paradigms = [api_schemas.ConjugationParadigm(**item) for item in paradigms_list]
        return paradigms

    async def generate_lemmas(
        self,
        lang: str,
        form: str,
        *,
        variety: Optional[str],
        source_preference: str,
        force_deterministic: bool,
    ) -> api_schemas.LemmaResponse:
        example = {
            "input": form,
            "candidates": [
                {
                    "lemma": "sleep",
                    "partofspeechcategory": "verb",
                    "confidence": 0.99,
                    "metadata": None,
                }
            ],
            "metadata": {
                "provenance": "llm",
                "confidence": 0.99,
                "variety": variety,
                "lemma_detected_from": None,
                "warnings": [],
                "notes": [],
            },
        }
        prompt = self._build_prompt(
            "Return lemma candidates for the given word, ordered by confidence.",
            {
                "lang": lang,
                "form": form,
                "variety": variety,
                "source_preference": source_preference,
                "force_deterministic": force_deterministic,
            },
            example=example,
        )
        data = await asyncio.to_thread(self._generate_json, prompt)
        self._ensure_list_field(data, field="candidates")
        self._ensure_metadata(data)
        for candidate in data["candidates"]:
            if isinstance(candidate, dict):
                candidate.setdefault("metadata", None)
        return api_schemas.LemmaResponse(**data)

    async def generate_pluralization(
        self,
        lang: str,
        form: str,
        *,
        target_number: str,
        gender: Optional[str],
        case: Optional[str],
        degree: Optional[str],
        variety: Optional[str],
        source_preference: str,
        force_deterministic: bool,
    ) -> api_schemas.InflectionResult:
        example = {
            "input": form,
            "results": ["example"],
            "features": {
                "partofspeechcategory": "noun",
                "number": target_number,
                "gender": gender,
                "case": case,
                "degree": degree,
            },
            "metadata": {
                "provenance": "llm",
                "confidence": 0.9,
                "variety": variety,
                "warnings": [],
                "notes": [],
            },
        }
        prompt = self._build_prompt(
            "Inflect the word for the requested number and any supplied grammatical features.",
            {
                "lang": lang,
                "form": form,
                "target_number": target_number,
                "gender": gender,
                "case": case,
                "degree": degree,
                "variety": variety,
                "source_preference": source_preference,
                "force_deterministic": force_deterministic,
            },
            example=example,
        )
        data = await asyncio.to_thread(self._generate_json, prompt)
        self._ensure_list_field(data, field="results", fallback_field="result")
        self._ensure_metadata(data)
        return api_schemas.InflectionResult(**data)

    async def generate_number_spelling(
        self,
        lang: str,
        digits: str,
        *,
        variety: Optional[str],
    ) -> api_schemas.NumberSpelling:
        example = {
            "input": digits,
            "text": "forty-two",
            "metadata": {
                "provenance": "llm",
                "confidence": 0.95,
                "variety": variety,
                "warnings": [],
                "notes": [],
            },
        }
        prompt = self._build_prompt(
            "Spell out the digits as words in the target language.",
            {"lang": lang, "digits": digits, "variety": variety},
            example=example,
        )
        data = await asyncio.to_thread(self._generate_json, prompt)
        self._ensure_metadata(data)
        return api_schemas.NumberSpelling(**data)

    async def generate_definitions(
        self,
        src_lang: str,
        tgt_lang: str,
        word: str,
        *,
        variety: Optional[str],
    ) -> api_schemas.DefinitionResponse:
        example = {
            "input": word,
            "results": [
                {
                    "surfaceform": word,
                    "text": word,
                    "root": word,
                    "definition": "Concise definition here.",
                    "translations": [],
                    "partofspeech": {"partofspeechcategory": "noun"},
                    "metadata": None,
                }
            ],
            "metadata": {
                "provenance": "llm",
                "confidence": 0.8,
                "variety": variety,
                "warnings": [],
                "notes": [],
            },
        }
        prompt = self._build_prompt(
            "Provide dictionary-style definitions (and translations when src_lang != tgt_lang).",
            {
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "word": word,
                "variety": variety,
            },
            example=example,
        )
        data = await asyncio.to_thread(self._generate_json, prompt)
        self._ensure_list_field(data, field="results", fallback_field="entries")
        self._ensure_metadata(data)
        return api_schemas.DefinitionResponse(**data)
