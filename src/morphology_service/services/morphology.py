from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, TypeVar

from pydantic import BaseModel

from morphology_service.api.v1 import schemas as api_schemas
from morphology_service.core.config import Settings, get_settings
from morphology_service.services.llm_client import LLMClient
from morphology_service.services.rule_engine import RuleEngine


TModel = TypeVar("TModel", bound=BaseModel)


class DeterministicUnavailableError(RuntimeError):
    """Raised when deterministic fallback is required but unavailable."""


@dataclass
class MorphologyService:
    """Orchestrates morphology operations across LLM and rule-based backends."""

    llm_client: LLMClient
    rule_engine: RuleEngine
    settings: Settings

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "MorphologyService":
        settings = settings or get_settings()
        llm_client = LLMClient(settings=settings)
        rule_engine = RuleEngine()
        return cls(llm_client=llm_client, rule_engine=rule_engine, settings=settings)

    # ---------------------------------------------------------------------- Verbs

    async def conjugate(
        self,
        lang: str,
        form: str,
        *,
        lemma: Optional[str] = None,
        expand_compound: bool = True,
        variety: Optional[str] = None,
        max_variants: int = 1,
        return_features: Optional[List[str]] = None,
        source_preference: str = "hybrid",
        force_deterministic: bool = False,
    ) -> List[api_schemas.ConjugationParadigm]:
        """Generate verb conjugations for `form`."""
        if self._use_rule_backend(source_preference, force_deterministic):
            try:
                return self.rule_engine.conjugate(
                    lang=lang,
                    form=form,
                    lemma=lemma,
                    expand_compound=expand_compound,
                    variety=variety,
                )
            except NotImplementedError as exc:
                if force_deterministic:
                    raise DeterministicUnavailableError(
                        f"Deterministic conjugation unsupported for language '{lang}'."
                    ) from exc
                # fall back to LLM path

        try:
            llm_result = await self.llm_client.generate_conjugations(
                lang=lang,
                form=form,
                lemma=lemma,
                expand_compound=expand_compound,
                variety=variety,
                max_variants=max_variants,
                return_features=return_features,
                source_preference=source_preference,
                force_deterministic=force_deterministic,
            )
        except Exception:
            try:
                return self._promote_hybrid(
                    self.rule_engine.conjugate(lang, form, lemma=lemma, variety=variety)
                )
            except NotImplementedError:
                raise

        if self._should_fallback_to_rule(llm_result):
            try:
                return self._promote_hybrid(
                    self.rule_engine.conjugate(lang, form, lemma=lemma, variety=variety)
                )
            except NotImplementedError:
                return llm_result

        return llm_result

    async def lemmatize(
        self,
        lang: str,
        form: str,
        *,
        variety: Optional[str] = None,
        source_preference: str = "hybrid",
        force_deterministic: bool = False,
    ) -> api_schemas.LemmaResponse:
        if self._use_rule_backend(source_preference, force_deterministic):
            try:
                return self.rule_engine.lemmatize(lang=lang, form=form, variety=variety)
            except NotImplementedError as exc:
                if force_deterministic:
                    raise DeterministicUnavailableError(
                        f"Deterministic lemmatization unsupported for language '{lang}'."
                    ) from exc
                # fall back to LLM

        try:
            llm_result = await self.llm_client.generate_lemmas(
                lang=lang,
                form=form,
                variety=variety,
                source_preference=source_preference,
                force_deterministic=force_deterministic,
            )
        except Exception:
            try:
                return self._promote_response_hybrid(
                    self.rule_engine.lemmatize(lang=lang, form=form, variety=variety)
                )
            except NotImplementedError:
                raise

        if self._response_needs_fallback(llm_result.metadata):
            try:
                return self._promote_response_hybrid(
                    self.rule_engine.lemmatize(lang=lang, form=form, variety=variety)
                )
            except NotImplementedError:
                return llm_result

        return llm_result

    # ------------------------------------------------------------- Nouns/adjectives

    async def pluralize(
        self,
        lang: str,
        form: str,
        *,
        target_number: str = "plural",
        gender: Optional[str] = None,
        case: Optional[str] = None,
        degree: Optional[str] = None,
        variety: Optional[str] = None,
        source_preference: str = "hybrid",
        force_deterministic: bool = False,
    ) -> api_schemas.PluralResponse:
        if self._use_rule_backend(source_preference, force_deterministic):
            try:
                rule_result = self.rule_engine.inflect_number(lang, form, target_number)
            except NotImplementedError as exc:
                if force_deterministic:
                    raise DeterministicUnavailableError(
                        f"Deterministic inflection unsupported for language '{lang}'."
                    ) from exc
            else:
                return api_schemas.PluralResponse(**rule_result.model_dump())

        try:
            base = await self.llm_client.generate_pluralization(
                lang=lang,
                form=form,
                target_number=target_number,
                gender=gender,
                case=case,
                degree=degree,
                variety=variety,
                source_preference=source_preference,
                force_deterministic=force_deterministic,
            )
        except Exception:
            try:
                fallback = self.rule_engine.inflect_number(lang, form, target_number)
            except NotImplementedError:
                raise
            fallback = self._promote_response_hybrid(fallback)
            return api_schemas.PluralResponse(**fallback.model_dump())

        if self._response_needs_fallback(base.metadata):
            try:
                fallback = self.rule_engine.inflect_number(lang, form, target_number)
            except NotImplementedError:
                return api_schemas.PluralResponse(**base.model_dump())
            fallback = self._promote_response_hybrid(fallback)
            return api_schemas.PluralResponse(**fallback.model_dump())

        return api_schemas.PluralResponse(**base.model_dump())

    async def singularize(
        self,
        lang: str,
        form: str,
        *,
        variety: Optional[str] = None,
        source_preference: str = "hybrid",
        force_deterministic: bool = False,
    ) -> api_schemas.SingularResponse:
        return api_schemas.SingularResponse(
            **(
                await self.pluralize(
                    lang=lang,
                    form=form,
                    target_number="singular",
                    variety=variety,
                    source_preference=source_preference,
                    force_deterministic=force_deterministic,
                )
            ).model_dump()
        )

    # -------------------------------------------------------------------- Utilities

    async def spell_number(
        self,
        lang: str,
        digits: str,
        *,
        variety: Optional[str] = None,
    ) -> api_schemas.NumberSpelling:
        try:
            result = await self.llm_client.generate_number_spelling(lang=lang, digits=digits, variety=variety)
        except Exception:
            try:
                return self.rule_engine.number_to_words(lang, digits)
            except NotImplementedError:
                raise

        if self._response_needs_fallback(result.metadata):
            try:
                fallback = self.rule_engine.number_to_words(lang, digits)
            except NotImplementedError:
                return result
            fallback.metadata.provenance = api_schemas.Provenance.HYBRID
            return fallback

        return result

    async def definitions(
        self,
        src_lang: str,
        tgt_lang: str,
        word: str,
        *,
        variety: Optional[str] = None,
    ) -> api_schemas.DefinitionResponse:
        try:
            return await self.llm_client.generate_definitions(
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                word=word,
                variety=variety,
            )
        except Exception:
            return self.rule_engine.definitions(src_lang=src_lang, tgt_lang=tgt_lang, word=word)

    async def help_languages(self) -> api_schemas.LanguagesResponse:
        return api_schemas.LanguagesResponse(languages=self.rule_engine.list_languages())

    async def help_dictionaries(self) -> api_schemas.DictionariesResponse:
        return api_schemas.DictionariesResponse(dictionaries=self.rule_engine.list_dictionaries())

    async def help_catalog(self, catalog: str) -> api_schemas.CatalogResponse:
        catalog_map: Dict[str, List[str]] = {
            "partsofspeech": self.rule_engine.list_parts_of_speech(),
            "persons": self.rule_engine.list_persons(),
            "tenses": self.rule_engine.list_tenses(),
        }
        items = catalog_map.get(catalog, [])
        return api_schemas.CatalogResponse(items=items)

    # ------------------------------------------------------------------ Internals

    def _use_rule_backend(self, source_preference: str, force_deterministic: bool) -> bool:
        if force_deterministic:
            return True
        return source_preference == "rule"

    def _should_fallback_to_rule(self, paradigms: List[api_schemas.ConjugationParadigm]) -> bool:
        if not paradigms:
            return True
        for paradigm in paradigms:
            if self._response_needs_fallback(paradigm.metadata):
                return True
        return False

    def _response_needs_fallback(self, metadata: Optional[api_schemas.Metadata]) -> bool:
        if metadata is None:
            return True
        threshold = self.settings.hybrid_confidence_threshold
        if metadata.confidence is None:
            return True
        return metadata.confidence < threshold

    def _promote_hybrid(self, paradigms: List[api_schemas.ConjugationParadigm]) -> List[api_schemas.ConjugationParadigm]:
        for paradigm in paradigms:
            paradigm.metadata.provenance = api_schemas.Provenance.HYBRID
        return paradigms

    def _promote_response_hybrid(self, response: TModel) -> TModel:
        if hasattr(response, "metadata") and response.metadata:
            response.metadata.provenance = api_schemas.Provenance.HYBRID
        return response
