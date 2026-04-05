from __future__ import annotations

import os
from dataclasses import dataclass

import requests
import streamlit as st


@dataclass
class LLMSettings:
    api_key: str = ""
    model: str = ""
    temperature: float = 0.4
    timeout: int = 45


class LLMClient:
    def __init__(self) -> None:
        self.settings = LLMSettings(
            api_key=st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", ""),
            model=st.secrets.get("OPENAI_MODEL", "") or os.getenv("OPENAI_MODEL", ""),
            temperature=float(
                st.secrets.get("OPENAI_TEMPERATURE", os.getenv("OPENAI_TEMPERATURE", "0.4"))
            ),
            timeout=int(
                st.secrets.get("OPENAI_TIMEOUT", os.getenv("OPENAI_TIMEOUT", "45"))
            ),
        )
        self.last_error: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.settings.api_key and self.settings.model)

    def generate(self, system_prompt: str, user_prompt: str) -> str | None:
        self.last_error = ""

        if not self.enabled:
            self.last_error = "LLM desactivado: falta OPENAI_API_KEY o OPENAI_MODEL."
            return None

        url = "https://api.openai.com/v1/responses"
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
        }

        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.settings.timeout,
            )

            if resp.status_code >= 400:
                self.last_error = f"HTTP {resp.status_code}: {resp.text[:700]}"
                return None

            data = resp.json()

            texts: list[str] = []
            for item in data.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        texts.append(content.get("text", "").strip())

            final_text = "\n".join(t for t in texts if t).strip()
            if not final_text:
                self.last_error = "La API respondió, pero no devolvió texto utilizable."
                return None

            return final_text

        except Exception as exc:
            self.last_error = f"Excepción llamando a Responses API: {type(exc).__name__}: {exc}"
            return None
