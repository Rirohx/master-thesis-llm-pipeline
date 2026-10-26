"""LLM client for an OpenAI-compatible endpoint."""

from __future__ import annotations
import json
import logging
import time
from typing import Any

import requests


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, config: dict, api_key: str):
        api_key = (api_key or "").strip()
        if not api_key:
            raise ValueError("api_key must be a non-empty")

        endpoint = str(config.get("endpoint", "")).strip()
        if not endpoint:
            raise ValueError("config.endpoint must be set")

        model = str(config.get("model", "")).strip()
        if not model:
            raise ValueError("config.model must be set")

        params = config.get("parameters", {})
        retry = config.get("retry", {})

        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.temperature = float(params.get("temperature", 0.4))
        self.top_p = float(params.get("top_p", 0.9))
        self.max_tokens = params.get("max_tokens")  # None means no cap
        self.max_retries = int(retry.get("max_retries", 3))
        self.backoff_factor = float(retry.get("backoff_factor", 2))
        self.timeout = float(config.get("timeout_seconds", 120))
        self._log = logging.getLogger("pipeline")

    def call(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty")

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if resp.status_code == 429 or resp.status_code >= 500:
                    raise LLMError(f"Transient HTTP {resp.status_code}: {resp.text[:300]}")
                if resp.status_code >= 400:
                    raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")

                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                if not isinstance(text, str) or not text.strip():
                    raise LLMError("Empty response content")

                return text

            except (requests.Timeout, requests.ConnectionError) as exc:
                last_err = exc
                if attempt < self.max_retries:
                    delay = self.backoff_factor ** (attempt + 1)
                    self._log.warning(
                        "Network error (attempt %d/%d): %s - retrying in %.1fs",
                        attempt + 1, self.max_retries + 1, exc, delay,
                    )
                    time.sleep(delay)
                    continue
                break

            except LLMError as exc:
                last_err = exc
                transient = "Transient" in str(exc) or "Empty" in str(exc)
                if attempt < self.max_retries and transient:
                    delay = self.backoff_factor ** (attempt + 1)
                    self._log.warning(
                        "LLM error (attempt %d/%d): %s - retrying in %.1fs",
                        attempt + 1, self.max_retries + 1, exc, delay,
                    )
                    time.sleep(delay)
                    continue
                break

        raise LLMError(
            f"LLM call failed after {self.max_retries + 1} attempts: {last_err}"
        ) from last_err

    @staticmethod
    def parse_json_array(text: str) -> list[Any]:
        text = text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"Invalid JSON in model response: {exc}\nRaw response: {text[:200]}"
            ) from exc
        if not isinstance(parsed, list):
            raise LLMError(
                f"Response is not a JSON array, got {type(parsed).__name__}"
            )
        return parsed
