from fastapi import FastAPI

from morphology_service.api.v1.routes import router as api_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Morphology API",
        version="0.1.0",
        description=(
            "LLM-backed morphology service inspired by Ultralingua's ULAPI. "
            "Supports conjugation, lemmatization, and inflection endpoints."
        ),
    )
    app.include_router(api_router, prefix="/v1")
    return app


app = create_app()
