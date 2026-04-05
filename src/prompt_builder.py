from __future__ import annotations

from typing import Any


def build_system_prompt() -> str:
    return (
        "Eres NeuroGuía, un sistema conversacional híbrido de apoyo no clínico. "
        "Responde siempre en español con un tono humano, cálido, claro y útil. "
        "No diagnostiques, no prescribas medicación, no indiques terapia específica "
        "y no sustituyas profesionales de salud. "
        "Tu prioridad es ayudar sin confundir más. "
        "Si ya hay suficiente contexto, evita seguir preguntando lo mismo y ofrece orientación concreta. "
        "Adapta tu lenguaje al perfil del usuario: "
        "si es docente, responde con foco en aula y manejo práctico; "
        "si es madre, padre o cuidador(a), combina contención emocional con acciones realistas; "
        "si es abuelo(a), usa un tono más afectivo, sencillo y cercano. "
        "No hables como formulario, no suenes a manual frío y no repitas estructuras. "
        "Cuando des pasos, ordénalos de forma breve, accionable y fácil de aplicar. "
        "Si el usuario expresa riesgo alto, prioriza seguridad presencial inmediata."
    )


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
    recent_dialogue = "\n".join(
        f"{role}: {content}" for role, content in previous_messages[-4:]
    )

    memory_text = "\n".join(
        f"- {m.get('categoria', 'general')}: {m.get('valor', '')}" for m in memory_items[:6]
    ) or "- sin memoria relevante"

    docs_text = "\n".join(
        f"- {d.get('title', 'sin título')}: {d.get('content', '')}" for d in retrieved_docs[:4]
    ) or "- sin conocimiento recuperado"

    profile_text = "\n".join(
        [
            f"- rol: {user_profile.get('role', 'no definido')}",
            f"- nombre preferido: {user_profile.get('display_name', '') or 'no indicado'}",
            f"- estado o región: {user_profile.get('state', '') or 'no indicado'}",
            f"- red de apoyo: {user_profile.get('support_network', '') or 'no indicado'}",
        ]
    )

    analysis_text = "\n".join(
        [
            f"- tema: {analysis.get('topic', 'desconocido')}",
            f"- intención: {analysis.get('intent', 'desconocida')}",
            f"- emoción dominante: {analysis.get('emotion', 'acompañamiento')}",
            f"- crisis: {analysis.get('crisis', 'sin_crisis')}",
            f"- límite clínico: {analysis.get('clinical_boundary', 'permitida')}",
            f"- contexto suficiente: {analysis.get('enough_context', False)}",
            f"- profundidad deseada: {analysis.get('depth', 'media')}",
        ]
    )

    return f"""Perfil del usuario:
{profile_text}

Análisis interno:
{analysis_text}

Memoria relevante:
{memory_text}

Conocimiento curado recuperado:
{docs_text}

Diálogo reciente:
{recent_dialogue or "- inicio de conversación"}

Consulta actual del usuario:
{user_message}

Instrucciones de salida:
- Responde en tono cercano y humano.
- Si el perfil es docente, da orientación práctica para aula.
- Si el perfil es abuelo(a), usa un tono más afectivo y sencillo.
- Si el perfil es madre/padre/cuidador(a), combina contención y acciones.
- Si ya hay suficiente contexto, NO pidas más datos antes de ayudar.
- Da entre 3 y 6 pasos concretos cuando haga falta.
- Cierra con una sola pregunta útil solo si realmente aporta valor.
"""
