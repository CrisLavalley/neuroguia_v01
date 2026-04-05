from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import streamlit as st

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class LLMSettings:
    api_key: str = ""
    model: str = "gpt-4.1-nano"
    temperature: float = 0.2
    timeout: int = 20
    max_output_tokens: int = 160
    cache_ttl_seconds: int = 604800
    daily_budget_usd: float = 1.0
    estimated_input_cost_per_1m: float = 0.10
    estimated_output_cost_per_1m: float = 0.40


class LLMClient:
    def __init__(self) -> None:
        self.settings = LLMSettings(
            api_key=st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", ""),
            model=st.secrets.get("OPENAI_MODEL", "") or os.getenv("OPENAI_MODEL", "gpt-4.1-nano"),
            temperature=float(st.secrets.get("OPENAI_TEMPERATURE", os.getenv("OPENAI_TEMPERATURE", "0.2"))),
            timeout=int(st.secrets.get("OPENAI_TIMEOUT", os.getenv("OPENAI_TIMEOUT", "20"))),
            max_output_tokens=int(st.secrets.get("OPENAI_MAX_OUTPUT_TOKENS", os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "160"))),
            cache_ttl_seconds=int(st.secrets.get("OPENAI_CACHE_TTL_SECONDS", os.getenv("OPENAI_CACHE_TTL_SECONDS", "604800"))),
            daily_budget_usd=float(st.secrets.get("OPENAI_DAILY_BUDGET_USD", os.getenv("OPENAI_DAILY_BUDGET_USD", "1.0"))),
            estimated_input_cost_per_1m=float(st.secrets.get("OPENAI_INPUT_COST_PER_1M", os.getenv("OPENAI_INPUT_COST_PER_1M", "0.10"))),
            estimated_output_cost_per_1m=float(st.secrets.get("OPENAI_OUTPUT_COST_PER_1M", os.getenv("OPENAI_OUTPUT_COST_PER_1M", "0.40"))),
        )
        self.last_error: str = ""
        self.last_meta: dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.settings.api_key and self.settings.model)

    def estimate_tokens(self, text: str) -> int:
        return max(1, int(len((text or "").strip()) / 4))

    def estimate_cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = (input_tokens / 1_000_000) * self.settings.estimated_input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * self.settings.estimated_output_cost_per_1m
        return round(input_cost + output_cost, 8)

    def _today_path(self) -> Path:
        return CACHE_DIR / f"cost_{time.strftime('%Y-%m-%d')}.json"

    def get_today_costs(self) -> dict[str, Any]:
        p = self._today_path()
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"date": time.strftime("%Y-%m-%d"), "estimated_cost_usd": 0.0, "calls": 0, "cache_hits": 0}

    def _save_today_costs(self, payload: dict[str, Any]) -> None:
        self._today_path().write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def can_spend(self, projected_cost: float) -> bool:
        data = self.get_today_costs()
        return (data.get("estimated_cost_usd", 0.0) + projected_cost) <= self.settings.daily_budget_usd

    def _cache_key(self, system_prompt: str, user_prompt: str) -> str:
        blob = json.dumps(
            {"model": self.settings.model, "system": system_prompt, "user": user_prompt, "max": self.settings.max_output_tokens},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return CACHE_DIR / f"{key}.json"

    def get_cache(self, system_prompt: str, user_prompt: str) -> str | None:
        p = self._cache_path(self._cache_key(system_prompt, user_prompt))
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if (time.time() - data.get("created_at", 0)) > self.settings.cache_ttl_seconds:
                return None
            costs = self.get_today_costs()
            costs["cache_hits"] = costs.get("cache_hits", 0) + 1
            self._save_today_costs(costs)
            self.last_meta = {"source": "cache", "estimated_cost_usd": 0.0}
            return data.get("response")
        except Exception:
            return None

    def save_cache(self, system_prompt: str, user_prompt: str, response: str, input_tokens: int, output_tokens: int) -> None:
        p = self._cache_path(self._cache_key(system_prompt, user_prompt))
        p.write_text(
            json.dumps(
                {
                    "created_at": time.time(),
                    "response": response,
                    "estimated_input_tokens": input_tokens,
                    "estimated_output_tokens": output_tokens,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str | None:
        self.last_error = ""
        self.last_meta = {}

        if not self.enabled:
            self.last_error = "LLM desactivado: falta OPENAI_API_KEY u OPENAI_MODEL."
            return None

        cached = self.get_cache(system_prompt, user_prompt)
        if cached:
            return cached

        input_tokens = self.estimate_tokens(system_prompt) + self.estimate_tokens(user_prompt)
        projected_cost = self.estimate_cost_usd(input_tokens, self.settings.max_output_tokens)

        if not self.can_spend(projected_cost):
            self.last_error = "Presupuesto diario estimado agotado."
            self.last_meta = {"source": "budget_guard", "estimated_cost_usd": projected_cost}
            return None

        url = "https://api.openai.com/v1/responses"
        headers = {"Authorization": f"Bearer {self.settings.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "max_output_tokens": self.settings.max_output_tokens,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.settings.timeout)
            if resp.status_code >= 400:
                self.last_error = f"HTTP {resp.status_code}: {resp.text[:350]}"
                self.last_meta = {"source": "api_error", "estimated_cost_usd": projected_cost}
                return None

            data = resp.json()
            texts = []
            for item in data.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        texts.append(content.get("text", "").strip())

            final_text = "\n".join(t for t in texts if t).strip()
            if not final_text:
                self.last_error = "La API respondió sin texto utilizable."
                self.last_meta = {"source": "empty_response", "estimated_cost_usd": projected_cost}
                return None

            output_tokens = self.estimate_tokens(final_text)
            real_cost = self.estimate_cost_usd(input_tokens, output_tokens)
            data_today = self.get_today_costs()
            data_today["estimated_cost_usd"] = round(data_today.get("estimated_cost_usd", 0.0) + real_cost, 8)
            data_today["calls"] = data_today.get("calls", 0) + 1
            self._save_today_costs(data_today)

            self.save_cache(system_prompt, user_prompt, final_text, input_tokens, output_tokens)
            self.last_meta = {
                "source": "llm",
                "estimated_input_tokens": input_tokens,
                "estimated_output_tokens": output_tokens,
                "estimated_cost_usd": real_cost,
            }
            return final_text
        except Exception as exc:
            self.last_error = f"Excepción en Responses API: {type(exc).__name__}: {exc}"
            self.last_meta = {"source": "exception", "estimated_cost_usd": projected_cost}
            return None
