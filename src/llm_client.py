from __future__ import annotations

import os
from dataclasses import dataclass

import requests


@dataclass
class LLMSettings:
    api_key: str = ""
    model: str = ""
    temperature: float = 0.4
    timeout: int = 45


class LLMClient:
    def __init__(self) -> None:
        self.settings = LLMSettings(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", ""),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.4")),
            timeout=int(os.getenv("OPENAI_TIMEOUT", "45")),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.settings.api_key and self.settings.model)

    def generate(self, system_prompt: str, user_prompt: str) -> str | None:
        if not self.enabled:
            return None

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.settings.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return None
