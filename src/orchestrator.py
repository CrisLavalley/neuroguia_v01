from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.llm_client import LLMClient
from src.prompt_builder import build_system_prompt, build_user_prompt, build_session_summary

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
        self.minimal_messages = {"si", "sí", "ok", "vale", "gracias", "ajá", "aja", "entiendo", "y entonces?", "y entonces", "mmm"}

    def build_welcome_message(self, display_name: str = "", role: str = "", user_memory: list[dict[str, Any]] | None = None) -> str:
        memory = user_memory or []
        temas = [m.get("valor", "") for m in memory if m.get("categoria") == "tema_frecuente"]
        memory_note = ""
        if temas:
            tema = self._topic_to_human(Counter(temas).most_common(1)[0][0])
            memory_note = f" También recuerdo que antes apareció el tema de **{tema}**."
        saludo = f"Hola, {display_name}." if display_name else "Hola."
        role_phrase = self._role_phrase(role)
        return (
            f"{saludo} Estoy aquí para acompañarte{role_phrase} con claridad y calidez.{memory_note}\n\n"
            "Puedes escribirme una situación concreta, una duda o algo que te esté pesando hoy. Voy a intentar responderte de forma útil, breve y aterrizada."
        )

    def analyze(self, message: str, context_dict: dict[str, Any] | None = None, user_memory: list[dict[str, Any]] | None = None,
                user_profile: dict[str, Any] | None = None, previous_messages: list[tuple[str, str]] | None = None) -> AnalysisResult:
        ctx = SessionContext(**(context_dict or {}))
        text = (message or "").strip()
        lowered = text.lower().strip()
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

        response, source = self._generate_cost_optimized_response(
            text=text,
            lowered=lowered,
            profile=profile,
            memory=memory,
            rag_hits=rag_hits,
            analysis=analysis,
            previous_messages=previous_messages,
        )

        new_ctx = self._update_context(ctx, text, topic, intent, enough_context, profile, analysis, memory)
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
            recurso_rag=rag_hits[:1],
            context=new_ctx,
            memory_updates=memory_updates,
        )

    def _should_call_llm(self, lowered: str, analysis: dict[str, Any], ctx: SessionContext) -> bool:
        if lowered in self.minimal_messages:
            return False
        if len(lowered.split()) <= 2:
            return False
        if analysis["clinical_boundary"] == "prohibida" or analysis["crisis"] == "riesgo_alto":
            return False
        if not analysis["enough_context"] and analysis["intent"] not in {"comprension", "acompanamiento_emocional"}:
            return False
        return True

    def _generate_cost_optimized_response(self, *, text: str, lowered: str, profile: dict[str, Any], memory: list[dict[str, Any]],
                                          rag_hits: list[dict[str, Any]], analysis: dict[str, Any],
                                          previous_messages: list[tuple[str, str]]) -> tuple[str, str]:
        if analysis["clinical_boundary"] == "prohibida":
            return (
                "Te leo con cuidado. No puedo diagnosticar, indicar medicación ni sustituir a profesionales de salud. Sí puedo ayudarte a ordenar lo que observas y a preparar una explicación clara para consulta.",
                "limite_no_clinico",
            )

        if analysis["crisis"] == "riesgo_alto":
            return (
                "Lo que describes suena a una situación de alto riesgo. Ahora lo más seguro es buscar apoyo presencial inmediato con una persona adulta responsable o un servicio de emergencia.",
                "alerta_riesgo",
            )

        if not self._should_call_llm(lowered, analysis, SessionContext()):
            return self._local_short_response(lowered, profile.get("role", ""), analysis), "local_short"

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
                meta = self.llm.last_meta or {}
                return llm_text.strip(), f"llm:{self.llm.settings.model} | costo≈${meta.get('estimated_cost_usd', 0.0):.6f}"

        fallback = self._deterministic_fallback(lowered=lowered, topic=analysis["topic"], intent=analysis["intent"], role=profile.get("role", ""), enough_context=analysis["enough_context"])
        reason = self.llm.last_error or "sin detalle"
        meta = self.llm.last_meta or {}
        return fallback, f"fallback_local ({reason}) | costo_est≈${meta.get('estimated_cost_usd', 0.0):.6f}"

    def _local_short_response(self, lowered: str, role: str, analysis: dict[str, Any]) -> str:
        if lowered in {"gracias", "ok", "vale", "sí", "si"}:
            return "Claro. Voy contigo. Si quieres, ahora lo traduzco a algo más concreto."
        if lowered in {"y entonces?", "y entonces"}:
            return "Lo aterrizo: dime qué te urge más, entender lo que pasa o saber qué hacer primero."
        if role == "docente":
            return "Voy contigo. Cuéntame en una sola frase qué pasó en el aula y qué quieres lograr, y te doy pasos concretos."
        return "Te sigo. Dime qué te gustaría resolver primero y lo bajamos a algo breve y útil."

    def _deterministic_fallback(self, *, lowered: str, topic: str, intent: str, role: str, enough_context: bool) -> str:
        opening = self._empathetic_opening(lowered, role)
        if enough_context and topic == "escuela_inclusiva":
            return f"{opening}\n\nMientras recuperamos la capa avanzada, te dejo un paso breve: baja estímulos, usa una instrucción corta, evita corregir en público y observa qué detonó el momento."
        if enough_context and topic == "vinculo_familiar":
            return f"{opening}\n\nComo orientación breve: empieza con momentos pequeños, evita el reproche directo y acércate desde algo que a ellos les interese."
        if enough_context and topic == "sueno":
            return f"{opening}\n\nComo primer paso, observa qué ocurre en la hora previa a dormir: pantallas, ruido, ansiedad o sobrecarga."
        return f"{opening}\n\nEstoy respondiendo en modo local austero. Puedo acompañarte brevemente, pero la versión más elaborada depende de la capa LLM."

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
        if text in self.minimal_messages or len(text.split()) <= 4:
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
        score = 0.28
        for token in ["urgente", "muy mal", "demasiado", "me duele", "me lastima", "berrinche"]:
            if token in text:
                score += 0.08
        return min(0.92, round(score, 2))

    def _estimate_confidence(self, topic: str, intent: str) -> float:
        score = 0.55 + (0.14 if topic != "acompanamiento_general" else 0) + (0.07 if intent != "acompanamiento" else 0)
        return min(0.9, round(score, 2))

    def _needs_clinical_boundary(self, text: str) -> bool:
        return any(k in text for k in self.boundary_patterns)

    def _select_protocol(self, topic: str, clinical: str) -> str:
        if clinical == "prohibida":
            return "protocolo_seguridad_clinica"
        mapping = {"sueno": "protocolo_sueno", "escuela_inclusiva": "protocolo_escuela_inclusiva", "vinculo_familiar": "protocolo_vinculo_familiar"}
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
        return [doc for _, doc in hits[:1]]

    def _has_enough_context(self, text: str, ctx: SessionContext, topic: str, intent: str) -> bool:
        if len(text.split()) >= 7:
            return True
        if ctx.turns >= 1 and topic == ctx.last_topic:
            return True
        if intent == "orientacion_practica" and len(text.split()) >= 5:
            return True
        return False

    def _role_phrase(self, role: str) -> str:
        mapping = {"madre": " como madre", "padre": " como padre", "abuelo(a)": " desde tu lugar de abuelo o abuela", "cuidador(a)": " desde tu papel de cuidado", "docente": " desde tu lugar docente"}
        return mapping.get(role, "")

    def _topic_to_human(self, topic: str) -> str:
        mapping = {"vinculo_familiar": "vínculo familiar", "escuela_inclusiva": "escuela", "sueno": "sueño"}
        return mapping.get(topic, topic.replace("_", " "))

    def _empathetic_opening(self, text: str, role: str) -> str:
        if role == "docente":
            return "Entiendo. En el aula, una situación así puede desgastar mucho."
        if any(x in text for x in ["me duele", "triste", "me preocupa"]):
            return "Te leo con cuidado. Eso puede pesar bastante."
        return "Gracias por abrir esto. Lo tomo con cuidado."

    def _update_context(self, ctx: SessionContext, text: str, topic: str, intent: str, enough_context: bool,
                        profile: dict[str, Any], analysis: dict[str, Any], memory: list[dict[str, Any]]) -> SessionContext:
        details = list(ctx.details_known)
        if text and text not in details:
            details.append(text[:120])

        summary = build_session_summary(user_profile=profile, analysis=analysis, memory_items=memory)

        return SessionContext(
            summary=summary,
            last_topic=topic,
            last_intent=intent,
            turns=ctx.turns + 1,
            case_active=True,
            details_known=details[-4:],
            emotional_tone=self._detect_emotion(text.lower()),
            last_user_need=intent,
            last_response_style="accion" if enough_context else "exploracion",
            enough_context=enough_context,
        )

    def _propose_memory_updates(self, text: str, topic: str, user_profile: dict[str, Any]) -> dict[str, Any]:
        updates = {"items": []}
        if topic != "acompanamiento_general":
            updates["items"].append(
                {"categoria": "tema_frecuente", "clave": "tema_frecuente", "valor": topic, "fuente": "conversacion", "nivel_confianza": 0.8}
            )
        return updates


_ENGINE = None


def get_ng_engine() -> NeuroGuiaEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = NeuroGuiaEngine()
    return _ENGINE
