from __future__ import annotations

from typing import Any


def compact_text(text: str, max_chars: int = 280) -> str:
    text = " ".join((text or "").split())
    return text[:max_chars].strip()


def build_system_prompt() -> str:
    return (
        "Eres NeuroGuía. Responde en español. Tono humano, cálido, claro y útil. "
        "No diagnostiques, no mediques, no indiques terapia específica y no sustituyas profesionales. "
        "Si ya hay contexto suficiente, no pidas más datos antes de ayudar. "
        "Da solo lo esencial: una explicación breve y luego 3 a 5 acciones cortas. "
        "No repitas frases, no suenes a manual y no alargues la respuesta."
    )


def build_session_summary(*, user_profile: dict[str, Any], analysis: dict[str, Any], memory_items: list[dict[str, Any]]) -> str:
    memory_values = []
    for item in memory_items[:2]:
        value = item.get("valor", "")
        if value:
            memory_values.append(value)
    summary = (
        f"rol={user_profile.get('role', 'no definido')}; "
        f"tema={analysis.get('topic', 'desconocido')}; "
        f"intencion={analysis.get('intent', 'desconocida')}; "
        f"emocion={analysis.get('emotion', 'acompanamiento')}; "
        f"memoria={', '.join(memory_values) if memory_values else 'sin_memoria'}"
    )
    return compact_text(summary, max_chars=320)


def build_user_prompt(
    *,
    user_message: str,
    user_profile: dict[str, Any],
    analysis: dict[str, Any],
    memory_items: list[dict[str, Any]],
    retrieved_docs: list[dict[str, Any]],
    previous_messages: list[tuple[str, str]] | None = None,
) -> str:
    previous_messages = previous_messages or []
    previous_assistant = ""
    for role, content in reversed(previous_messages[-2:]):
        if role == "assistant":
            previous_assistant = compact_text(content, max_chars=120)
            break

    docs_text = "sin_apoyo_curado"
    if retrieved_docs:
        d = retrieved_docs[0]
        docs_text = f"{d.get('title', 'Protocolo')}: {compact_text(d.get('content', ''), 140)}"

    summary = build_session_summary(
        user_profile=user_profile,
        analysis=analysis,
        memory_items=memory_items,
    )

    return f"""Resumen:
{summary}

Ultima respuesta:
{previous_assistant or 'sin_respuesta_previa'}

Apoyo curado:
{docs_text}

Consulta:
{compact_text(user_message, 420)}

Salida:
- Respuesta breve.
- Maximo 5 pasos.
- No repitas la pregunta.
- Aterriza algo accionable.
"""
