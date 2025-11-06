from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from morphology_service.api.v1 import schemas
from morphology_service.services.morphology import DeterministicUnavailableError, MorphologyService

CONJUGATION_EXAMPLE = [
    {
        "infinitive": "sleep",
        "conjugations": [
            {
                "surfaceform": "sleep",
                "text": "sleep",
                "root": "sleep",
                "partofspeech": {"partofspeechcategory": "verb", "tense": "infinitive"},
                "metadata": None,
            },
            {
                "surfaceform": "sleeps",
                "text": "sleeps",
                "root": "sleep",
                "partofspeech": {
                    "partofspeechcategory": "verb",
                    "tense": "present",
                    "person": "third",
                    "number": "singular",
                },
                "metadata": None,
            },
            {
                "surfaceform": "slept",
                "text": "slept",
                "root": "sleep",
                "partofspeech": {"partofspeechcategory": "verb", "tense": "past"},
                "metadata": None,
            },
            {
                "surfaceform": "sleeping",
                "text": "sleeping",
                "root": "sleep",
                "partofspeech": {"partofspeechcategory": "verb", "tense": "presentparticiple"},
                "metadata": None,
            },
        ],
        "metadata": {
            "provenance": "llm",
            "confidence": 0.92,
            "variety": "en-GB",
            "lemma_detected_from": "slept",
            "warnings": [],
            "notes": [],
        },
    }
]

LEMMA_EXAMPLE = {
    "input": "slept",
    "candidates": [
        {"lemma": "sleep", "partofspeechcategory": "verb", "confidence": 0.99, "metadata": None}
    ],
    "metadata": {
        "provenance": "llm",
        "confidence": 0.99,
        "variety": "en-GB",
        "lemma_detected_from": None,
        "warnings": [],
        "notes": [],
    },
}

PLURAL_EXAMPLE = {
    "input": "mouse",
    "results": ["mice"],
    "features": {"partofspeechcategory": "noun", "number": "plural"},
    "metadata": {"provenance": "llm", "confidence": 0.95, "variety": None, "warnings": [], "notes": []},
}

SINGULAR_EXAMPLE = {
    "input": "mice",
    "results": ["mouse"],
    "features": {"partofspeechcategory": "noun", "number": "singular"},
    "metadata": {"provenance": "llm", "confidence": 0.95, "variety": None, "warnings": [], "notes": []},
}

NUMBER_EXAMPLE = {
    "input": "4321",
    "text": "four thousand, three hundred and twenty-one",
    "metadata": {"provenance": "llm", "confidence": 0.9, "variety": "en-GB", "warnings": [], "notes": []},
}

DEFINITION_EXAMPLE = {
    "input": "mouse",
    "results": [
        {
            "surfaceform": "mouse",
            "text": "mouse",
            "root": "mouse",
            "definition": "A small rodent with a pointed snout and a long tail.",
            "translations": ["ratón"],
            "partofspeech": {"partofspeechcategory": "noun"},
            "metadata": None,
        }
    ],
    "metadata": {"provenance": "llm", "confidence": 0.85, "variety": None, "warnings": [], "notes": []},
}


router = APIRouter()


async def get_service() -> MorphologyService:
    return MorphologyService.from_settings()


@router.get(
    "/conjugations/{lang}/{form}",
    response_model=List[schemas.ConjugationParadigm],
    summary="Get verb conjugations",
    description="Return the full conjugation paradigm for the supplied verb form.",
    responses={
        200: {"description": "Verb conjugations returned", "content": {"application/json": {"example": CONJUGATION_EXAMPLE}}},
        422: {"description": "Deterministic mode requested but unavailable"},
    },
)
async def get_conjugations(
    lang: str = Path(..., example="en", description="BCP-47 or ISO language code"),
    form: str = Path(..., example="sleep", description="Surface form or lemma to conjugate"),
    lemma: Optional[str] = None,
    expand_compound: bool = Query(default=True),
    variety: Optional[str] = None,
    max_variants: int = Query(default=1, ge=1, le=5),
    return_features: Optional[List[str]] = Query(default=None),
    source_preference: str = Query(default="hybrid"),
    force_deterministic: bool = Query(default=False),
    service: MorphologyService = Depends(get_service),
):
    try:
        return await service.conjugate(
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
    except DeterministicUnavailableError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/lemmas/{lang}/{form}",
    response_model=schemas.LemmaResponse,
    summary="Lemmatize a form",
    description="Return lemma candidates and part of speech for the supplied form.",
    responses={
        200: {"description": "Lemma candidates returned", "content": {"application/json": {"example": LEMMA_EXAMPLE}}},
        422: {"description": "Deterministic mode requested but unavailable"},
    },
)
async def get_lemmas(
    lang: str = Path(..., example="en"),
    form: str = Path(..., example="slept"),
    variety: Optional[str] = None,
    source_preference: str = Query(default="hybrid"),
    force_deterministic: bool = Query(default=False),
    service: MorphologyService = Depends(get_service),
):
    try:
        return await service.lemmatize(
            lang=lang,
            form=form,
            variety=variety,
            source_preference=source_preference,
            force_deterministic=force_deterministic,
        )
    except DeterministicUnavailableError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/plurals/{lang}/{form}",
    response_model=schemas.PluralResponse,
    summary="Pluralize a form",
    description="Inflect a noun or adjective into the requested number.",
    responses={
        200: {"description": "Pluralization result", "content": {"application/json": {"example": PLURAL_EXAMPLE}}},
        422: {"description": "Deterministic mode requested but unavailable"},
    },
)
async def get_plurals(
    lang: str = Path(..., example="en"),
    form: str = Path(..., example="mouse"),
    target_number: str = Query(default="plural"),
    gender: Optional[str] = None,
    case: Optional[str] = None,
    degree: Optional[str] = None,
    variety: Optional[str] = None,
    source_preference: str = Query(default="hybrid"),
    force_deterministic: bool = Query(default=False),
    service: MorphologyService = Depends(get_service),
):
    try:
        return await service.pluralize(
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
    except DeterministicUnavailableError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/singulars/{lang}/{form}",
    response_model=schemas.SingularResponse,
    summary="Singularize a form",
    description="Return singular candidates for a plural input.",
    responses={
        200: {"description": "Singularization result", "content": {"application/json": {"example": SINGULAR_EXAMPLE}}},
        422: {"description": "Deterministic mode requested but unavailable"},
    },
)
async def get_singulars(
    lang: str = Path(..., example="en"),
    form: str = Path(..., example="mice"),
    variety: Optional[str] = None,
    source_preference: str = Query(default="hybrid"),
    force_deterministic: bool = Query(default=False),
    service: MorphologyService = Depends(get_service),
):
    try:
        return await service.singularize(
            lang=lang,
            form=form,
            variety=variety,
            source_preference=source_preference,
            force_deterministic=force_deterministic,
        )
    except DeterministicUnavailableError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/numbers/{lang}/{digits}",
    response_model=schemas.NumberSpelling,
    summary="Spell out digits",
    description="Convert digits into words in the requested language.",
    responses={
        200: {"description": "Number spelled out", "content": {"application/json": {"example": NUMBER_EXAMPLE}}},
    },
)
async def get_numbers(
    lang: str = Path(..., example="en"),
    digits: str = Path(..., example="4321"),
    variety: Optional[str] = None,
    service: MorphologyService = Depends(get_service),
):
    return await service.spell_number(lang=lang, digits=digits, variety=variety)


@router.get(
    "/definitions/{src_lang}/{tgt_lang}/{word}",
    response_model=schemas.DefinitionResponse,
    summary="Lookup definitions or translations",
    description="Return dictionary-style entries for the supplied word.",
    responses={
        200: {"description": "Dictionary entries", "content": {"application/json": {"example": DEFINITION_EXAMPLE}}},
    },
)
async def get_definitions(
    src_lang: str = Path(..., example="en"),
    tgt_lang: str = Path(..., example="es"),
    word: str = Path(..., example="mouse"),
    variety: Optional[str] = None,
    service: MorphologyService = Depends(get_service),
):
    return await service.definitions(src_lang=src_lang, tgt_lang=tgt_lang, word=word, variety=variety)


@router.get(
    "/terms/{src_lang}/{tgt_lang}/{word}",
    response_model=schemas.DefinitionResponse,
    summary="Lookup multi-word terms",
    description="Return phrase-level dictionary entries for the supplied term.",
)
async def get_terms(
    src_lang: str = Path(..., example="en"),
    tgt_lang: str = Path(..., example="fr"),
    word: str = Path(..., example="break a leg"),
    variety: Optional[str] = None,
    service: MorphologyService = Depends(get_service),
):
    return await service.definitions(src_lang=src_lang, tgt_lang=tgt_lang, word=word, variety=variety)


@router.get("/help/languages", response_model=schemas.LanguagesResponse)
async def list_languages(service: MorphologyService = Depends(get_service)):
    return await service.help_languages()


@router.get("/help/dictionaries", response_model=schemas.DictionariesResponse)
async def list_dictionaries(service: MorphologyService = Depends(get_service)):
    return await service.help_dictionaries()


@router.get("/help/{catalog}", response_model=schemas.CatalogResponse)
async def list_catalog(catalog: str, service: MorphologyService = Depends(get_service)):
    if catalog not in {"partsofspeech", "persons", "tenses"}:
        raise HTTPException(status_code=404, detail="Catalog not found")
    return await service.help_catalog(catalog)
