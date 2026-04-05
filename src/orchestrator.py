from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.llm_client import LLMClient
from src.prompt_builder import build_system_prompt, build_user_prompt

BASE = Path(__file__).resolve().parents[1]
KB_DIR = BASE / "knowledge_base"


def _safe_load_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


@dataclass
class SessionContext:
    summary: str = ""
    last_topic: str = ""
    last_intent: str = ""
    turns: int = 0
    user_goal: str = ""
    case_active: bool = False
    details_known: list[str] = field(default_factory=list)
    emotional_tone: str = ""
    last_user_need: str = ""
    last_response_style: str = ""
    enough_context: bool = False


@dataclass
class AnalysisResult:
    emocion: str
    probabilidad: float
    intensidad: float
    crisis_tipo: str
    filtro_clinico: str
    protocolo: str
    respuesta: str
    fuente_respuesta: str
    intent: str
    topic: str
    recurso_rag: list[dict[str, Any]]
    context: SessionContext
    memory_updates: dict[str, Any]


class NeuroGuiaEngine:
    def __init__(self) -> None:
        self.rag_docs = _safe_load_json(KB_DIR / "rag_documents.json", [])
        self.protocols = _safe_load_json(KB_DIR / "protocolos_intervencion.json", {})
        self.llm = LLMClient()
        self.boundary_patterns = [
            "diagnóstico", "diagnostico", "medicación", "medicacion", "medicamento",
            "receta", "dosis", "terapia específica", "terapia especifica", "pastilla",
            "fármaco", "farmaco", "tratamiento", "medicar", "psiquiatra", "neurólogo",
            "neurologo", "antidepresivo", "ansiolítico", "ansiolitico"
        ]
        self.high_risk_patterns = [
            "suicid", "se quiere hacer daño", "se quiere lastimar", "arma",
            "convuls", "no respira", "me quiero morir", "quitarme la vida"
        ]

    def build_welcome_message(
        self,
        display_name: str = "",
        role: str = "",
        user_memory: list[dict[str, Any]] | None = None,
    ) -> str:
        memory = user_memory or []
        temas = [m.get("valor", "") for m in memory if m.get("categoria") == "tema_frecuente"]
        memory_note = ""
        if temas:
            tema = self._topic_to_human(Counter(temas).most_common(1)[0][0])
            memory_note = f" También recuerdo que antes apareció el tema de **{tema}**."

        saludo = f"Hola, {display_name}." if display_name else "Hola."
        role_phrase = self._role_phrase(role)

        return (
            f"{saludo} Estoy aquí para acompañarte{role_phrase} con un tono claro, cercano y útil.{memory_note}\n\n"
            "Puedes escribirme una situación concreta, una duda, algo que te dolió hoy o una pregunta general. "
            "Mi intención es ayudarte con calidez y, cuando ya haya suficiente contexto, con orientación realmente concreta."
        )

    def analyze(
        self,
        message: str,
        context_dict: dict[str, Any] | None = None,
        user_memory: list[dict[str, Any]] | None = None,
        user_profile: dict[str, Any] | None = None,
        previous_messages: list[tuple[str, str]] | None = None,
    ) -> AnalysisResult:
        ctx = SessionContext(**(context_dict or {}))
        text = (message or "").strip()
        lowered = text.lower()
        memory = user_memory or []
        profile = user_profile or {}
        previous_messages = previous_messages or []

        topic = self._detect_topic(lowered, ctx, memory)
        intent = self._detect_intent(lowered, ctx)
        emotion = self._detect_emotion(lowered)
        crisis = self._detect_crisis(lowered)
        intensity = self._estimate_intensity(lowered)
        confidence = self._estimate_confidence(topic, intent)
        clinical = "prohibida" if self._needs_clinical_boundary(lowered) else "permitida"
        enough_context = self._has_enough_context(lowered, ctx, topic, intent)
        depth = self._depth_by_role(profile.get("role", ""), lowered, ctx)
        rag_hits = self._retrieve_rag(lowered, topic)

        analysis = {
            "topic": topic,
            "intent": intent,
            "emotion": emotion,
            "crisis": crisis,
            "clinical_boundary": clinical,
            "enough_context": enough_context,
            "depth": depth,
        }

        response, source = self._generate_hybrid_response(
            text=text,
            lowered=lowered,
            profile=profile,
            memory=memory,
            rag_hits=rag_hits,
            analysis=analysis,
            previous_messages=previous_messages,
        )

        new_ctx = self._update_context(ctx, text, topic, intent, enough_context)
        memory_updates = self._propose_memory_updates(lowered, topic, profile)

        return AnalysisResult(
            emocion=emotion,
            probabilidad=confidence,
            intensidad=intensity,
            crisis_tipo=crisis,
            filtro_clinico=clinical,
            protocolo=self._select_protocol(topic, clinical),
            respuesta=response,
            fuente_respuesta=source,
            intent=intent,
            topic=topic,
            recurso_rag=rag_hits[:3],
            context=new_ctx,
            memory_updates=memory_updates,
        )

    def _generate_hybrid_response(
        self,
        *,
        text: str,
        lowered: str,
        profile: dict[str, Any],
        memory: list[dict[str, Any]],
        rag_hits: list[dict[str, Any]],
        analysis: dict[str, Any],
        previous_messages: list[tuple[str, str]],
    ) -> tuple[str, str]:
        if analysis["clinical_boundary"] == "prohibida":
            return (
                "Te leo con cuidado. No puedo diagnosticar, indicar medicación ni sustituir a profesionales de salud. "
                "Sí puedo ayudarte a ordenar lo que observas, identificar detonantes, preparar preguntas seguras y pensar cómo acompañar de forma prudente.",
                "limite_no_clinico",
            )

        if analysis["crisis"] == "riesgo_alto":
            return (
                "Lo que describes suena a una situación de alto riesgo. En este momento lo más seguro es buscar apoyo presencial inmediato "
                "con una persona adulta responsable o un servicio de emergencia. Quédate cerca, reduce riesgos del entorno y prioriza ayuda directa.",
                "alerta_riesgo",
            )

        if self.llm.enabled:
            system_prompt = build_system_prompt()
            user_prompt = build_user_prompt(
                user_message=text,
                user_profile=profile,
                analysis=analysis,
                memory_items=memory,
                retrieved_docs=rag_hits,
                previous_messages=previous_messages,
            )
            llm_text = self.llm.generate(system_prompt, user_prompt)
            if llm_text:
                return llm_text.strip(), f"llm:{self.llm.settings.model}"

        fallback = self._deterministic_fallback(
            lowered=lowered,
            topic=analysis["topic"],
            intent=analysis["intent"],
            role=profile.get("role", ""),
            enough_context=analysis["enough_context"],
            depth=analysis["depth"],
        )
        reason = self.llm.last_error or "sin detalle"
        return fallback, f"fallback_local ({reason})"

    def _deterministic_fallback(
        self,
        *,
        lowered: str,
        topic: str,
        intent: str,
        role: str,
        enough_context: bool,
        depth: str,
    ) -> str:
        opening = self._empathetic_opening(lowered, role)

        if enough_context and topic == "sueno":
            steps = [
                "observa a qué hora empieza realmente la dificultad",
                "revisa qué pasó en la hora previa: pantallas, ruido, ansiedad, conflicto o sobrecarga",
                "mantén una rutina breve y repetible antes de dormir",
                "anota durante varios días si notas un patrón repetido",
            ]
            return (
                f"{opening}\n\n"
                "Puedo darte una orientación inicial mientras dejamos estable la conexión con el modelo.\n\n"
                + self._format_steps(steps, depth)
                + "\n\nSi quieres, después lo convertimos en un registro simple y más útil para casa."
            )

        if enough_context and topic == "escuela_inclusiva":
            steps = [
                "en el momento del desborde, baja estímulos y protege al grupo antes de corregir",
                "usa una instrucción corta y repetible, no un sermón",
                "reduce público y exposición si el alumno ya está muy activado",
                "observa qué pasó justo antes para detectar detonantes",
            ]
            return (
                f"{opening}\n\n"
                "Puedo darte una orientación inicial mientras dejamos estable la conexión con el modelo.\n\n"
                + self._format_steps(steps, depth)
                + "\n\nEn cuanto el modelo quede entrando bien, esta parte debe volverse mucho más específica y natural."
            )

        return (
            f"{opening}\n\n"
            "Estoy teniendo que responder con una capa local más simple de lo ideal. "
            "Puedo darte una orientación básica, pero la idea es que esto mejore en cuanto la conexión con el modelo quede estable."
        )

    def _detect_topic(self, text: str, ctx: SessionContext, memory: list[dict[str, Any]]) -> str:
        if any(x in text for x in ["escuela", "maestra", "docente", "clase", "tarea", "alumno", "compañeros", "companeros"]):
            return "escuela_inclusiva"
        if any(x in text for x in ["duerme", "sueño", "sueno", "insomnio", "noche"]):
            return "sueno"
        if any(x in text for x in ["acercarme", "empatizar", "conectar", "nietos", "hijos", "cariñosos", "carinosos", "vínculo", "vinculo"]):
            return "vinculo_familiar"

        temas = [m.get("valor", "") for m in memory if m.get("categoria") == "tema_frecuente"]
        if temas:
            return Counter(temas).most_common(1)[0][0]
        return ctx.last_topic or "acompanamiento_general"

    def _detect_intent(self, text: str, ctx: SessionContext) -> str:
        if self._needs_clinical_boundary(text):
            return "consulta_clinica"
        if any(x in text for x in ["cómo", "como", "qué hago", "que hago", "qué puedo hacer", "que puedo hacer"]):
            return "orientacion_practica"
        if any(x in text for x in ["por qué", "por que", "quisiera entender", "quiero entender", "no entiendo"]):
            return "comprension"
        if any(x in text for x in ["me duele", "me siento", "me preocupa"]):
            return "acompanamiento_emocional"
        if text in ["sí", "si", "ok", "vale", "aja", "ajá", "bien"] or len(text.split()) <= 4:
            return "seguimiento" if ctx.case_active else "acompanamiento"
        return "acompanamiento"

    def _detect_emotion(self, text: str) -> str:
        if any(x in text for x in ["preocupada", "preocupado", "miedo", "ansiedad", "ansiosa", "ansioso"]):
            return "ansiedad"
        if any(x in text for x in ["triste", "dolor", "me duele", "me lastima"]):
            return "tristeza"
        return "acompanamiento"

    def _detect_crisis(self, text: str) -> str:
        if any(x in text for x in self.high_risk_patterns):
            return "riesgo_alto"
        return "sin_crisis"

    def _estimate_intensity(self, text: str) -> float:
        score = 0.30
        for token in ["urgente", "muy mal", "demasiado", "me duele", "me lastima", "berrinche"]:
            if token in text:
                score += 0.08
        return min(0.95, round(score, 2))

    def _estimate_confidence(self, topic: str, intent: str) -> float:
        score = 0.56 + (0.14 if topic != "acompanamiento_general" else 0) + (0.09 if intent != "acompanamiento" else 0)
        return min(0.95, round(score, 2))

    def _needs_clinical_boundary(self, text: str) -> bool:
        return any(k in text for k in self.boundary_patterns)

    def _select_protocol(self, topic: str, clinical: str) -> str:
        if clinical == "prohibida":
            return "protocolo_seguridad_clinica"
        mapping = {
            "sueno": "protocolo_sueno",
            "escuela_inclusiva": "protocolo_escuela_inclusiva",
            "vinculo_familiar": "protocolo_vinculo_familiar",
        }
        return mapping.get(topic, "protocolo_acompanamiento")

    def _retrieve_rag(self, text: str, topic: str) -> list[dict[str, Any]]:
        hits = []
        for doc in self.rag_docs:
            score = 3 if doc.get("topic") == topic else 0
            for kw in doc.get("keywords", []):
                if kw in text:
                    score += 1
            if score > 0:
                hits.append((score, doc))
        hits.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in hits[:3]]

    def _has_enough_context(self, text: str, ctx: SessionContext, topic: str, intent: str) -> bool:
        if len(text.split()) >= 8:
            return True
        if ctx.turns >= 1 and topic == ctx.last_topic:
            return True
        if intent == "orientacion_practica" and len(text.split()) >= 5:
            return True
        return False

    def _depth_by_role(self, role: str, text: str, ctx: SessionContext) -> str:
        if role in ["madre", "padre", "docente", "cuidador(a)"]:
            return "alto" if ctx.turns >= 1 else "medio"
        return "medio"

    def _format_steps(self, steps: list[str], depth: str) -> str:
        selected = steps[:4] if depth != "alto" else steps
        return "\n".join(f"{i+1}. {step}" for i, step in enumerate(selected))

    def _role_phrase(self, role: str) -> str:
        mapping = {
            "madre": " como madre",
            "padre": " como padre",
            "abuelo(a)": " desde tu lugar de abuelo o abuela",
            "cuidador(a)": " desde tu papel de cuidado",
            "docente": " desde tu lugar docente",
        }
        return mapping.get(role, "")

    def _topic_to_human(self, topic: str) -> str:
        mapping = {
            "vinculo_familiar": "vínculo familiar",
            "escuela_inclusiva": "escuela",
            "sueno": "sueño",
        }
        return mapping.get(topic, topic.replace("_", " "))

    def _empathetic_opening(self, text: str, role: str) -> str:
        if role == "docente":
            return "Entiendo. En el aula, una situación así puede desgastar mucho."
        if any(x in text for x in ["me duele", "triste", "me preocupa"]):
            return "Te leo con cuidado. Eso puede pesar bastante."
        return "Gracias por abrir esto. Lo tomo con cuidado."

    def _update_context(self, ctx: SessionContext, text: str, topic: str, intent: str, enough_context: bool) -> SessionContext:
        details = list(ctx.details_known)
        if text and text not in details:
            details.append(text[:140])
        return SessionContext(
            summary=f"tema={topic}; intent={intent}; detalles={len(details)}",
            last_topic=topic,
            last_intent=intent,
            turns=ctx.turns + 1,
            case_active=True,
            details_known=details[-6:],
            emotional_tone=self._detect_emotion(text.lower()),
            last_user_need=intent,
            last_response_style="accion" if enough_context else "exploracion",
            enough_context=enough_context,
        )

    def _propose_memory_updates(self, text: str, topic: str, user_profile: dict[str, Any]) -> dict[str, Any]:
        updates = {"items": []}
        if topic != "acompanamiento_general":
            updates["items"].append(
                {
                    "categoria": "tema_frecuente",
                    "clave": "tema_frecuente",
                    "valor": topic,
                    "fuente": "conversacion",
                    "nivel_confianza": 0.8,
                }
            )
        return updates


_ENGINE = None


def get_ng_engine() -> NeuroGuiaEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = NeuroGuiaEngine()
    return _ENGINE
