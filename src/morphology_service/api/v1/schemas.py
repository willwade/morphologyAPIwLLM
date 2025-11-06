from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Provenance(str, Enum):
    LLM = "llm"
    RULE = "rule"
    HYBRID = "hybrid"


class PartOfSpeechFeatures(BaseModel):
    partofspeechcategory: Optional[str] = Field(default=None)
    pospsubcategory: Optional[str] = Field(default=None)
    tense: Optional[str] = Field(default=None)
    mood: Optional[str] = Field(default=None)
    aspect: Optional[str] = Field(default=None)
    voice: Optional[str] = Field(default=None)
    person: Optional[str] = Field(default=None)
    number: Optional[str] = Field(default=None)
    gender: Optional[str] = Field(default=None)
    polarity: Optional[str] = Field(default=None)
    case: Optional[str] = Field(default=None)
    degree: Optional[str] = Field(default=None)
    form: Optional[str] = Field(default=None)


class Metadata(BaseModel):
    provenance: Provenance
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    variety: Optional[str] = None
    lemma_detected_from: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class MorphologyItem(BaseModel):
    surfaceform: str
    text: Optional[str] = None
    root: Optional[str] = None
    partofspeech: PartOfSpeechFeatures
    metadata: Optional[Metadata] = None


class ConjugationParadigm(BaseModel):
    infinitive: str
    conjugations: List[MorphologyItem]
    metadata: Metadata


class LemmaCandidate(BaseModel):
    lemma: str
    partofspeechcategory: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata: Optional[Metadata] = None


class LemmaResponse(BaseModel):
    input: str
    candidates: List[LemmaCandidate]
    metadata: Metadata


class InflectionResult(BaseModel):
    input: str
    results: List[str]
    features: PartOfSpeechFeatures
    metadata: Metadata


class PluralResponse(InflectionResult):
    pass


class SingularResponse(InflectionResult):
    pass


class NumberSpelling(BaseModel):
    input: str
    text: str
    metadata: Metadata


class DefinitionItem(MorphologyItem):
    definition: Optional[str] = None
    translations: Optional[List[str]] = None


class DefinitionResponse(BaseModel):
    input: str
    results: List[DefinitionItem]
    metadata: Metadata


class BatchItemRequest(BaseModel):
    word: str


class BatchConjugationRequest(BaseModel):
    items: List[str]
    options: Optional[dict] = None


class HelpEntry(BaseModel):
    id: str
    name: str
    description: Optional[str] = None


class LanguagesResponse(BaseModel):
    languages: List[dict]


class DictionariesResponse(BaseModel):
    dictionaries: List[dict]


class CatalogResponse(BaseModel):
    items: List[str]
