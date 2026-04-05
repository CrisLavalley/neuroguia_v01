from __future__ import annotations

from typing import Any


def compact_text(text: str, max_chars: int = 320) -> str:
    text = " ".join((text or "").split())
    return text[:max_chars].strip()


def build_system_prompt() -> str:
    return (
        "Eres NeuroGuía. Responde en español, con tono humano, claro, cálido y útil. "
        "No diagnostiques, no mediques, no indiques terapia específica y no sustituyas profesionales de salud. "
        "Si ya hay suficiente contexto, no hagas más preguntas antes de ayudar. "
        "Da solo lo esencial: explica brevemente el porqué y luego aterriza la acción. "
        "Usa 1 párrafo corto o de 3 a 5 pasos breves. "
        "No repitas frases, no suenes a manual y no alargues la respuesta. "
        "Si el perfil es docente, responde en clave de aula. "
        "Si es madre, padre o cuidador(a), combina contención y acciones. "
        "Si es abuelo(a), usa un tono más afectivo y sencillo."
    )


def build_session_summary(
    *,
    user_profile: dict[str, Any],
    analysis: dict[str, Any],
    memory_items: list[dict[str, Any]],
) -> str:
    memory_values = []
    for item in memory_items[:2]:
        categoria = item.get("categoria", "general")
        valor = item.get("valor", "")
        if valor:
            memory_values.append(f"{categoria}:{valor}")

    summary = (
        f"rol={user_profile.get('role', 'no definido')}; "
        f"tema={analysis.get('topic', 'desconocido')}; "
        f"intención={analysis.get('intent', 'desconocida')}; "
        f"emoción={analysis.get('emotion', 'acompañamiento')}; "
        f"contexto_suficiente={analysis.get('enough_context', False)}; "
        f"memoria={', '.join(memory_values) if memory_values else 'sin_memoria'}"
    )
    return compact_text(summary, max_chars=380)


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
    last_assistant = ""
    for role, content in reversed(previous_messages[-2:]):
        if role == "assistant":
            last_assistant = compact_text(content, max_chars=160)
            break

    summary = build_session_summary(
        user_profile=user_profile,
        analysis=analysis,
        memory_items=memory_items,
    )

    docs_min = []
    for d in retrieved_docs[:1]:
        title = d.get("title", "sin título")
        content = compact_text(d.get("content", ""), max_chars=180)
        docs_min.append(f"{title}: {content}")
    docs_text = docs_min[0] if docs_min else "sin_apoyo_curado"

    return f"""Resumen:
{summary}

Última respuesta del sistema:
{last_assistant or 'sin_respuesta_previa'}

Apoyo curado:
{docs_text}

Consulta actual:
{compact_text(user_message, max_chars=500)}

Reglas de salida:
- Respuesta breve.
- Máximo 5 pasos.
- No repitas la pregunta.
- No uses frases genéricas repetitivas.
- Ayuda con algo accionable y humano.
"""
