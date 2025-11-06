import asyncio
import os
from typing import Any, Dict, Tuple

import httpx

BASE_URL = os.environ.get("MORPHOLOGY_BASE_URL", "http://127.0.0.1:8000/v1")


async def fetch_json(client: httpx.AsyncClient, path: str, params: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    response = await client.get(f"{BASE_URL}{path}", params=params, timeout=60.0)
    data = response.json()
    return response.status_code, data


async def main() -> None:
    tests = [
        (
            "/conjugations/en/slept",
            {"expand_compound": False, "source_preference": "llm"},
            lambda data: data and data[0]["metadata"]["provenance"] == "llm",
        ),
        (
            "/lemmas/en/slept",
            {"source_preference": "llm"},
            lambda data: data["metadata"]["provenance"] == "llm",
        ),
        (
            "/plurals/en/mouse",
            {"target_number": "plural", "source_preference": "llm"},
            lambda data: data["metadata"]["provenance"] == "llm",
        ),
    ]

    async with httpx.AsyncClient() as client:
        for path, params, predicate in tests:
            status, payload = await fetch_json(client, path, params)
            provenance = None
            if isinstance(payload, list) and payload:
                provenance = payload[0].get("metadata", {}).get("provenance")
            elif isinstance(payload, dict):
                provenance = payload.get("metadata", {}).get("provenance")

            print(f"{path} -> status {status}, provenance={provenance}")
            if status != 200:
                print("  ❌ Request failed:", payload)
            elif not predicate(payload):
                print("  ⚠️ Response did not come from LLM, payload:", payload)
            else:
                print("  ✅ LLM backend responded successfully.")


if __name__ == "__main__":
    asyncio.run(main())
