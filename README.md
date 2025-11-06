# Morphology API (LLM-backed)

FastAPI service that provides morphology endpoints (conjugations, lemmas, plurals, number spelling, etc.). The primary backend uses [Simon Willison's `llm`](https://github.com/simonw/llm) library so you can swap between Gemini, Ollama, OpenAI, or any other supported provider. A lightweight deterministic rule engine ships with the service for offline or forced-deterministic flows: We want to extend this to HFSTs in the future so less is by LLM and more by these. 

## Features

- `/v1/conjugations/{lang}/{form}` – full verb paradigms with metadata
- `/v1/lemmas/{lang}/{form}` – lemma candidates + POS
- `/v1/plurals/{lang}/{form}`, `/v1/singulars/{lang}/{form}` – noun/adj inflection
- `/v1/numbers/{lang}/{digits}` – number → words
- `/v1/definitions`, `/v1/terms` – dictionary-style lookups (LLM-driven today)
- `/v1/help/*` – catalogs mirroring ULAPI (`partsofspeech`, `tenses`, etc.)
- Hybrid orchestration: LLM first, deterministic rule engine fallback on low confidence
- English deterministic backend implemented with HFST finite-state transducers (falls back gracefully if HFST is unavailable)
- `force_deterministic=true` shortcut for rule-only responses (returns HTTP 422 if unavailable)

## Getting Started

```bash
pipx install uv  # one-time, if uv is not already available
uv venv
source .venv/bin/activate
uv pip install -e '.[dev]'
```

Configure your preferred LLM provider. For Gemini:

```bash
export GEMINI_API_KEY="AIza..."
uv run llm install llm-gemini  # installs the Gemini plugin once per environment
```

Prefer not to keep secrets in your shell history? Drop the key into a file and point the service at it:

```bash
mkdir -p .secrets
echo "AIza..." > .secrets/gemini.key
echo "MORPHOLOGY_GEMINI_API_KEY_FILE=.secrets/gemini.key" >> .env
```

The settings loader will read the key on startup and expose it to the `llm` plugin automatically.

### HFST finite-state support

The deterministic English backend now prefers [HFST](https://hfst.github.io) transducers. The Python bindings are pulled in automatically through `hfst` when you install the project dependencies. On platforms without prebuilt HFST wheels, the service will fall back to the legacy heuristic rules while still exposing the same API surface. To force HFST usage, ensure the package installs cleanly in your environment:

```bash
uv pip install hfst
```

At runtime you can confirm the backend selection via `/v1/help/languages`, which now annotates each language with its deterministic provider.

Run the API:

```bash
uv run uvicorn morphology_service.main:app --reload
```

Example request:

```bash
curl "http://localhost:8000/v1/conjugations/en/slept?expand_compound=true&variety=en-GB"
```

## Configuration

Settings are loaded via environment variables using the `MORPHOLOGY_` prefix. Key options:

| Env var | Default | Description |
| --- | --- | --- |
| `MORPHOLOGY_LLM_MODEL_NAME` | `gemini-2.0-flash` | Model identifier passed to `llm.get_model` |
| `MORPHOLOGY_LLM_TEMPERATURE` | `0.0` | Temperature for generations |
| `MORPHOLOGY_HYBRID_CONFIDENCE_THRESHOLD` | `0.8` | Confidence cut-off before falling back to deterministic rules |
| `MORPHOLOGY_MAX_LLM_RETRIES` | `2` | Retries for transient LLM failures |

Optional `.env` files are supported.

## Tests

```bash
uv run pytest
```

`tests/` contains coverage around the hybrid orchestration and rule engine heuristics. Add your own gold lists for target languages as you plug in richer rule datasets.

## Extending

- Swap models by installing the relevant `llm-*` plugin and updating `MORPHOLOGY_LLM_MODEL_NAME`
- Implement richer deterministic engines via the `RuleEngine` class (e.g., integrate Pattern, Apertium)
- Add caching or persistence by wrapping calls in your storage layer of choice

## Roadmap Ideas

- JSON Schema validation + structured retries for LLM responses
- More granular provenance (per conjugated form)
- Batch endpoints (`/v1/batch/*`) for higher throughput
- Sandboxed offline bundle with Ollama or llama.cpp backends
