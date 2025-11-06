import asyncio

import pytest

from morphology_service.api.v1 import schemas
from morphology_service.core.config import Settings
from morphology_service.services.morphology import MorphologyService
from morphology_service.services.rule_engine import RuleEngine


class StubLLMClient:
    def __init__(self, conjugation_confidence: float = 0.5):
        self.conjugation_confidence = conjugation_confidence

    async def generate_conjugations(self, **_: object):
        metadata = schemas.Metadata(
            provenance=schemas.Provenance.LLM,
            confidence=self.conjugation_confidence,
        )
        return [
            schemas.ConjugationParadigm(
                infinitive="sleep",
                conjugations=[],
                metadata=metadata,
            )
        ]

    async def generate_lemmas(self, **_: object):
        return schemas.LemmaResponse(
            input="slept",
            candidates=[],
            metadata=schemas.Metadata(
                provenance=schemas.Provenance.LLM,
                confidence=0.5,
            ),
        )

    async def generate_pluralization(self, **_: object):
        return schemas.InflectionResult(
            input="mouse",
            results=["mouses"],
            features=schemas.PartOfSpeechFeatures(partofspeechcategory="noun", number="plural"),
            metadata=schemas.Metadata(
                provenance=schemas.Provenance.LLM,
                confidence=0.4,
            ),
        )

    async def generate_number_spelling(self, **_: object):
        return schemas.NumberSpelling(
            input="42",
            text="fourty two",
            metadata=schemas.Metadata(provenance=schemas.Provenance.LLM, confidence=0.4),
        )

    async def generate_definitions(self, **_: object):
        return schemas.DefinitionResponse(
            input="mouse",
            results=[],
            metadata=schemas.Metadata(provenance=schemas.Provenance.LLM, confidence=0.5),
        )


@pytest.mark.asyncio
async def test_hybrid_fallback_promotes_rule_metadata():
    service = MorphologyService(
        llm_client=StubLLMClient(conjugation_confidence=0.4),
        rule_engine=RuleEngine(),
        settings=Settings(),
    )

    paradigms = await service.conjugate("en", "slept")
    assert paradigms[0].metadata.provenance == schemas.Provenance.HYBRID
    assert paradigms[0].metadata.confidence == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_force_deterministic_uses_rule_engine():
    service = MorphologyService(
        llm_client=StubLLMClient(),
        rule_engine=RuleEngine(),
        settings=Settings(),
    )

    result = await service.pluralize("en", "mouse", force_deterministic=True)
    assert result.results == ["mice"]
    assert result.metadata.provenance == schemas.Provenance.RULE


def test_rule_engine_number_to_words():
    rule_engine = RuleEngine()
    spelling = rule_engine.number_to_words("en", "1234")
    assert "one thousand" in spelling.text
