
from __future__ import annotations

from typing import Any

import requests


def wikipedia_summary(query: str, lang: str = "es", timeout: int = 6) -> dict[str, Any] | None:
    """
    Recupera un resumen breve desde la API pública de Wikipedia.
    Se usa solo para dudas informativas no clínicas.
    """
    if not query or len(query.strip()) < 3:
        return None

    search_url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "utf8": 1,
    }

    try:
        resp = requests.get(search_url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("query", {}).get("search", [])
        if not results:
            return None

        title = results[0].get("title")
        if not title:
            return None

        summary_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"
        resp2 = requests.get(summary_url, timeout=timeout)
        resp2.raise_for_status()
        data2 = resp2.json()

        extract = data2.get("extract")
        page_url = data2.get("content_urls", {}).get("desktop", {}).get("page", "")
        if not extract:
            return None

        return {
            "title": title,
            "summary": extract,
            "url": page_url,
            "source": "Wikipedia",
        }
    except Exception:
        return None
