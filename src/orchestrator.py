from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.llm_client import LLMClient
from src.prompt_builder import build_system_prompt, build_user_prompt, build_session_summary, compact_text

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
    distilled_case: str = ""
    last_topic: str = ""
    last_intent: str = ""
    turns: int = 0
    llm_calls: int = 0
    enough_context: bool = False
    details_known: list[str] = field(default_factory=list)


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
        self.llm = LLMClient()
        self.protocol_bank = _safe_load_json(KB_DIR / "protocol_bank.json", [])
        self.minimal_messages = {"si", "sí", "ok", "vale", "gracias", "aja", "ajá", "y entonces", "y entonces?", "entiendo", "mmm"}
        self.boundary_patterns = ["diagnóstico","diagnostico","medicación","medicacion","medicamento","dosis","pastilla","tratamiento","medicar","psiquiatra","neurólogo","neurologo"]
        self.high_risk_patterns = ["suicid","me quiero morir","se quiere lastimar","arma","no respira","convuls"]

    def build_welcome_message(self, display_name: str = "", role: str = "", user_memory: list[dict[str, Any]] | None = None) -> str:
        saludo = f"Hola, {display_name}." if display_name else "Hola."
        role_phrase = {
            "madre": " como madre",
            "padre": " como padre",
            "docente": " desde tu lugar docente",
            "cuidador(a)": " desde tu papel de cuidado",
            "abuelo(a)": " desde tu lugar de abuelo o abuela",
        }.get(role, "")
        return (
            f"{saludo} Estoy aquí para acompañarte{role_phrase} con claridad y calidez.\n\n"
            "Puedes escribirme con libertad. Mi trabajo es ayudarte a ordenar lo que pasa y darte una orientación útil, breve y aterrizada."
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
        clinical = "prohibida" if self._needs_clinical_boundary(lowered) else "permitida"
        enough_context = self._has_enough_context(lowered, ctx, topic, intent)

        distilled_case = self._distill_user_message(text, topic, intent, emotion)
        protocol_hits = self._retrieve_protocols(lowered, topic)
        protocol_response = self._protocol_first_response(topic, profile.get("role", ""), protocol_hits)

        response, source, llm_used = self._resolve_response(
            text=text, lowered=lowered, topic=topic, intent=intent, emotion=emotion, clinical=clinical, crisis=crisis,
            profile=profile, memory=memory, previous_messages=previous_messages, ctx=ctx,
            protocol_hits=protocol_hits, protocol_response=protocol_response, distilled_case=distilled_case, enough_context=enough_context
        )

        summary = build_session_summary(
            user_profile=profile,
            analysis={"topic": topic, "intent": intent, "emotion": emotion, "enough_context": enough_context},
            memory_items=memory,
        )

        new_ctx = SessionContext(
            summary=summary,
            distilled_case=distilled_case,
            last_topic=topic,
            last_intent=intent,
            turns=ctx.turns + 1,
            llm_calls=ctx.llm_calls + (1 if llm_used else 0),
            enough_context=enough_context,
            details_known=(ctx.details_known + [compact_text(text, 100)])[-4:],
        )

        memory_updates = self._propose_memory_updates(topic)

        return AnalysisResult(
            emocion=emotion,
            probabilidad=0.78,
            intensidad=intensity,
            crisis_tipo=crisis,
            filtro_clinico=clinical,
            protocolo=protocol_hits[0]["id"] if protocol_hits else "protocolo_acompanamiento",
            respuesta=response,
            fuente_respuesta=source,
            intent=intent,
            topic=topic,
            recurso_rag=protocol_hits[:1],
            context=new_ctx,
            memory_updates=memory_updates,
        )

    def _resolve_response(self, *, text: str, lowered: str, topic: str, intent: str, emotion: str, clinical: str, crisis: str,
                          profile: dict[str, Any], memory: list[dict[str, Any]], previous_messages: list[tuple[str, str]],
                          ctx: SessionContext, protocol_hits: list[dict[str, Any]], protocol_response: str,
                          distilled_case: str, enough_context: bool) -> tuple[str, str, bool]:
        if clinical == "prohibida":
            return ("Te leo con cuidado. No puedo diagnosticar ni indicar medicación, pero sí ayudarte a ordenar lo que observas y preparar una explicación clara para consulta.", "limite_no_clinico", False)
        if crisis == "riesgo_alto":
            return ("Lo que describes suena a una situación de alto riesgo. Ahora lo más importante es buscar apoyo presencial inmediato con una persona adulta responsable o un servicio de emergencia.", "alerta_riesgo", False)
        if lowered in self.minimal_messages:
            return (self._local_followup(lowered), "local_followup", False)

        if protocol_response and topic != "acompanamiento_general":
            if self._should_call_llm(lowered, enough_context, ctx):
                system_prompt = build_system_prompt()
                user_prompt = build_user_prompt(
                    user_message=f"Caso destilado: {distilled_case}\n\nBase protocolaria: {protocol_response}",
                    user_profile=profile,
                    analysis={"topic": topic, "intent": intent, "emotion": emotion, "enough_context": enough_context},
                    memory_items=memory,
                    retrieved_docs=protocol_hits[:1],
                    previous_messages=previous_messages,
                )
                llm_text = self.llm.generate(system_prompt, user_prompt)
                if llm_text:
                    cost = self.llm.last_meta.get("estimated_cost_usd", 0.0)
                    return (llm_text.strip(), f"llm:{self.llm.settings.model} | costo≈${cost:.6f}", True)
            reason = self.llm.last_error or "sin_llm"
            return (protocol_response, f"protocol_first ({reason})", False)

        if self._should_call_llm(lowered, enough_context, ctx):
            system_prompt = build_system_prompt()
            user_prompt = build_user_prompt(
                user_message=distilled_case,
                user_profile=profile,
                analysis={"topic": topic, "intent": intent, "emotion": emotion, "enough_context": enough_context},
                memory_items=memory,
                retrieved_docs=[],
                previous_messages=previous_messages,
            )
            llm_text = self.llm.generate(system_prompt, user_prompt)
            if llm_text:
                cost = self.llm.last_meta.get("estimated_cost_usd", 0.0)
                return (llm_text.strip(), f"llm:{self.llm.settings.model} | costo≈${cost:.6f}", True)

        reason = self.llm.last_error or "modo_local"
        return (self._short_local_fallback(topic), f"fallback_local ({reason})", False)

    def _should_call_llm(self, lowered: str, enough_context: bool, ctx: SessionContext) -> bool:
        if len(lowered.split()) <= 2 or lowered in self.minimal_messages:
            return False
        if ctx.llm_calls >= 2:
            return False
        if not enough_context:
            return False
        return True

    def _local_followup(self, lowered: str) -> str:
        if lowered in {"gracias", "ok", "vale", "sí", "si"}:
            return "Claro. Sigo contigo. Si quieres, ahora lo aterrizo en algo todavía más concreto."
        return "Te sigo. Dime qué te urge más: entender lo que pasa o saber qué hacer primero."

    def _short_local_fallback(self, topic: str) -> str:
        if topic == "sueno":
            return "Como primer paso, observa qué ocurre en la hora previa a dormir: pantallas, ruido, ansiedad o sobrecarga. Si quieres, luego lo traducimos a una rutina breve."
        if topic == "vinculo_familiar":
            return "Como orientación breve: evita el reproche directo, acércate desde algo que sí les interese y busca momentos pequeños, predecibles y amables."
        if topic == "escuela_inclusiva":
            return "Como primer apoyo: baja estímulos, usa una instrucción corta, evita corregir en público y detecta qué detonó el momento."
        return "Puedo acompañarte con una orientación breve. Si me dices qué te preocupa más, lo bajo a algo práctico."

    def _distill_user_message(self, text: str, topic: str, intent: str, emotion: str) -> str:
        lowered = text.lower()
        conditions = []
        for token in ["tea","tdah","aacc","autismo","dislexia","dispraxia","discalculia","tourette","sensorial"]:
            if token in lowered:
                conditions.append(token.upper() if token in {"tea","tdah","aacc"} else token)
        situation = compact_text(text, 120)
        distilled = f"tema={topic}; intencion={intent}; emocion={emotion}; condiciones={', '.join(conditions) if conditions else 'no_identificadas'}; situacion={situation}"
        return compact_text(distilled, 220)

    def _retrieve_protocols(self, lowered: str, topic: str) -> list[dict[str, Any]]:
        hits = []
        for doc in self.protocol_bank:
            score = 3 if doc.get("topic") == topic else 0
            for kw in doc.get("keywords", []):
                if kw in lowered:
                    score += 1
            if score > 0:
                hits.append((score, doc))
        hits.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in hits[:2]]

    def _protocol_first_response(self, topic: str, role: str, protocol_hits: list[dict[str, Any]]) -> str:
        if not protocol_hits:
            return ""
        doc = protocol_hits[0]
        opening = {
            "docente": "Entiendo. En el aula, una situación así puede desgastar mucho.",
            "madre": "Te leo con cuidado. Esto puede doler y cansar a la vez.",
            "padre": "Te leo con cuidado. Esto puede doler y cansar a la vez.",
            "cuidador(a)": "Te leo con cuidado. Esto puede doler y cansar a la vez.",
            "abuelo(a)": "Te acompaño. Estas situaciones tocan fibras muy profundas.",
        }.get(role, "Gracias por abrir esto. Lo tomo con cuidado.")
        actions = doc.get("actions", [])[:4]
        action_lines = "\n".join(f"{i+1}. {a}" for i, a in enumerate(actions))
        why = doc.get("why_it_helps", "")
        closing = doc.get("closing", "")
        return f"{opening}\n\n{doc.get('lead', '')}\n\n{action_lines}\n\nPor qué ayuda: {why}\n\n{closing}".strip()

    def _detect_topic(self, text: str, ctx: SessionContext, memory: list[dict[str, Any]]) -> str:
        if any(x in text for x in ["escuela","maestra","docente","clase","tarea","salón","salon","alumno","compañeros","companeros"]):
            return "escuela_inclusiva"
        if any(x in text for x in ["duerme","sueño","sueno","insomnio","noche","dormir"]):
            return "sueno"
        if any(x in text for x in ["acercarme","empatizar","conectar","cariñosos","carinosos","afecto","vínculo","vinculo","relación","relacion","nietos","hijos"]):
            return "vinculo_familiar"
        if any(x in text for x in ["sensorial","ruido","luces","sobrecarga","estimulos","estímulos"]):
            return "sobrecarga_sensorial"
        if any(x in text for x in ["agotada","agotado","rebasada","rebasado","cansada","cansado","ya no puedo"]):
            return "cansancio_cuidador"
        temas = [m.get("valor", "") for m in memory if m.get("categoria") == "tema_frecuente"]
        if temas:
            return Counter(temas).most_common(1)[0][0]
        return ctx.last_topic or "acompanamiento_general"

    def _detect_intent(self, text: str, ctx: SessionContext) -> str:
        if self._needs_clinical_boundary(text):
            return "consulta_clinica"
        if any(x in text for x in ["cómo","como","qué hago","que hago","qué puedo hacer","que puedo hacer"]):
            return "orientacion_practica"
        if any(x in text for x in ["por qué","por que","quisiera entender","quiero entender","no entiendo"]):
            return "comprension"
        if any(x in text for x in ["me duele","me siento","me preocupa","me rebasa"]):
            return "acompanamiento_emocional"
        if text in self.minimal_messages or len(text.split()) <= 4:
            return "seguimiento" if ctx.turns > 0 else "acompanamiento"
        return "acompanamiento"

    def _detect_emotion(self, text: str) -> str:
        if any(x in text for x in ["preocupada","preocupado","miedo","ansiedad","ansiosa","ansioso"]):
            return "ansiedad"
        if any(x in text for x in ["triste","dolor","me duele","me lastima"]):
            return "tristeza"
        if any(x in text for x in ["agotada","agotado","rebasada","rebasado"]):
            return "agotamiento"
        return "acompanamiento"

    def _detect_crisis(self, text: str) -> str:
        if any(x in text for x in self.high_risk_patterns):
            return "riesgo_alto"
        return "sin_crisis"

    def _estimate_intensity(self, text: str) -> float:
        score = 0.30
        for token in ["urgente","muy mal","demasiado","me duele","ya no puedo","explota","grita"]:
            if token in text:
                score += 0.08
        return min(0.95, round(score, 2))

    def _needs_clinical_boundary(self, text: str) -> bool:
        return any(k in text for k in self.boundary_patterns)

    def _has_enough_context(self, text: str, ctx: SessionContext, topic: str, intent: str) -> bool:
        if len(text.split()) >= 7:
            return True
        if ctx.turns >= 1 and topic == ctx.last_topic:
            return True
        if intent == "orientacion_practica" and len(text.split()) >= 5:
            return True
        return False

    def _propose_memory_updates(self, topic: str) -> dict[str, Any]:
        items = []
        if topic != "acompanamiento_general":
            items.append({"categoria": "tema_frecuente", "clave": "tema_frecuente", "valor": topic, "fuente": "conversacion", "nivel_confianza": 0.8})
        return {"items": items}
