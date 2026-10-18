"""Client for the local vLLM server (OpenAI-compatible API) serving Qwen3-VL.

The server runs on the same host (127.0.0.1); frames never leave the machine. Images go as base64 JPEG data URLs.
"""
from __future__ import annotations

import asyncio
import base64
import io
import os

import httpx
import numpy as np
from PIL import Image

VLLM_URL = os.environ.get("FORTE_VLLM_URL", "http://127.0.0.1:8001/v1")
VLLM_MODEL = os.environ.get("FORTE_VLLM_MODEL", "qwen3vl")


def jpeg_b64(frame: np.ndarray, quality: int = 90, scale: float = 1.0) -> str:
    """RGB uint8 frame -> base64 JPEG."""
    im = Image.fromarray(np.ascontiguousarray(frame))
    if scale != 1.0:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class VlmClient:
    def __init__(self, base_url: str = VLLM_URL, model: str = VLLM_MODEL, concurrency: int = 32,
                 timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.concurrency = concurrency
        self.timeout = timeout

    def ready(self) -> bool:
        try:
            return httpx.get(f"{self.base_url}/models", timeout=5.0).status_code == 200
        except httpx.HTTPError:
            return False

    def ask_many(self, requests: list[dict]) -> list[str | None]:
        """Run chat requests concurrently; answers come back in input order, None where a request failed.

        Request keys: images (list of base64 JPEG), prompt (str), system (str, optional),
        schema (JSON schema dict, optional; turns on structured output), max_tokens (int, default 256).
        The user message holds the images first, then the prompt. Put long fixed instructions in `system`:
        it is the shared prefix that vLLM prefix caching reuses across requests.
        """
        return asyncio.run(self._ask_all(requests))

    async def _ask_all(self, requests: list[dict]) -> list[str | None]:
        sem = asyncio.Semaphore(self.concurrency)
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as http:
            async def one(req: dict) -> str | None:
                async with sem:
                    for attempt in range(3):
                        try:
                            r = await http.post("/chat/completions", json=self._body(req))
                            r.raise_for_status()
                            return r.json()["choices"][0]["message"]["content"]
                        except (httpx.HTTPError, KeyError, IndexError, ValueError):
                            if attempt == 2:
                                return None
                            await asyncio.sleep(1.0 + attempt)
                    return None

            return list(await asyncio.gather(*(one(q) for q in requests)))

    def _body(self, req: dict) -> dict:
        content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
                   for b in req.get("images", [])]
        content.append({"type": "text", "text": req["prompt"]})
        messages = [{"role": "system", "content": req["system"]}] if req.get("system") else []
        messages.append({"role": "user", "content": content})
        body = {"model": self.model, "messages": messages, "max_tokens": req.get("max_tokens", 256),
                "temperature": 0.0}
        if req.get("schema"):
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "answer", "schema": req["schema"]}}
        return body
