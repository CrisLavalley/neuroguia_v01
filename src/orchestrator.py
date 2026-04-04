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
        self.web_info_patterns = [
            "qué es", "que es", "explícame", "explicame", "información sobre",
            "informacion sobre", "háblame de", "hablame de", "quién es", "quien es",
            "qué significa", "que significa"
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
            "Yo voy a intentar responderte con calidez, sin meterte en un formato rígido y, cuando ya haya suficiente contexto, "
            "con pasos concretos."
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
        crisis = self._detect_crisis(lowered, topic)
        intensity = self._estimate_intensity(lowered)
        confidence = self._estimate_confidence(lowered, topic, intent)
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

        response = self._generate_hybrid_response(
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
        fuente = "motor_hibrido_llm" if self.llm.enabled else "motor_hibrido_fallback"

        return AnalysisResult(
            emocion=emotion,
            probabilidad=confidence,
            intensidad=intensity,
            crisis_tipo=crisis,
            filtro_clinico=clinical,
            protocolo=self._select_protocol(topic, clinical),
            respuesta=response,
            fuente_respuesta=fuente,
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
    ) -> str:
        if analysis["clinical_boundary"] == "prohibida":
            return (
                "Te leo con cuidado. No puedo diagnosticar ni indicar medicación o tratamiento. "
                "Sí puedo ayudarte a ordenar lo que observas, identificar detonantes, preparar preguntas para consulta "
                "y pensar cómo acompañar de forma segura."
            )

        if analysis["crisis"] == "riesgo_alto":
            return (
                "Lo que describes suena a una situación de alto riesgo. En este momento lo más seguro es buscar apoyo presencial inmediato "
                "con una persona adulta responsable o un servicio de emergencia. Quédate cerca, reduce riesgos del entorno y prioriza ayuda directa."
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
                return llm_text.strip()

        return self._deterministic_fallback(
            lowered=lowered,
            topic=analysis["topic"],
            intent=analysis["intent"],
            role=profile.get("role", ""),
            enough_context=analysis["enough_context"],
            depth=analysis["depth"],
        )

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
        opening = self._empathetic_opening(lowered, role, "")
        if enough_context:
            if topic == "escuela_inclusiva":
                plan = [
                    "En el momento del desborde, evita razonar o regañar; primero baja estímulos y protege al grupo.",
                    "Usa una instrucción corta y repetible, por ejemplo: “vamos a calmarnos aquí” o “primero respiramos y luego hablamos”.",
                    "Si se tira al piso o hace berrinche, reduce público y exposición; mientras menos espectadores, mejor.",
                    "Observa señales previas como ruido, frustración, cambios de rutina o conflictos con compañeros.",
                    "Prepara una salida preventiva: una pausa breve, un espacio tranquilo o una rutina clara antes de que escale.",
                ]
                return (
                    f"{opening}\n\n"
                    "Con lo que me dices, ya hay suficiente contexto para orientarte de forma concreta.\n\n"
                    + self._format_steps(plan, depth)
                    + "\n\nNo lo leas solo como mala conducta; muchas veces hay desregulación, sobrecarga o dificultad para sostener el contexto. "
                    "Si quieres, te ayudo a convertir esto en una estrategia breve para usar mañana en clase."
                )

            if topic == "vinculo_familiar":
                plan = [
                    "No busques cercanía desde el reproche; eso suele cerrar más la puerta.",
                    "Empieza con momentos pequeños y naturales, no con conversaciones demasiado grandes.",
                    "Acércate desde algo que a ellos les interese, aunque parezca simple.",
                    "Haz comentarios o invitaciones ligeras en lugar de interrogatorios.",
                    "Valora cualquier gesto pequeño de conexión, porque el vínculo suele reconstruirse poco a poco.",
                ]
                return (
                    f"{opening}\n\n"
                    "Con lo que me cuentas, lo más útil es reconstruir el vínculo sin presión.\n\n"
                    + self._format_steps(plan, depth)
                    + "\n\nSi quieres, puedo ayudarte a pensar una frase o actividad concreta según la edad de tus hijos o nietos."
                )

            if topic == "sueno":
                plan = [
                    "Observa a qué hora empieza realmente la dificultad.",
                    "Revisa qué pasó en la hora previa: pantallas, ruido, ansiedad, conflicto o sobrecarga.",
                    "Mantén una rutina breve y repetible antes de dormir.",
                    "Anota durante algunos días si hay patrón o detonante repetido.",
                ]
                return (
                    f"{opening}\n\n"
                    "Con esto ya se puede trabajar de forma práctica.\n\n"
                    + self._format_steps(plan, depth)
                    + "\n\nSi quieres, te ayudo a crear un registro muy simple para observarlo mejor."
                )

        if intent == "orientacion_practica":
            return (
                f"{opening}\n\n"
                "Para darte una orientación útil, dime en una sola frase qué pasó y qué te gustaría lograr, y con eso te doy pasos concretos."
            )

        return (
            f"{opening}\n\n"
            "Gracias por contármelo. Si quieres, dime qué te gustaría que cambiara primero y lo aterrizamos en acciones concretas."
        )

    def _detect_topic(self, text: str, ctx: SessionContext, memory: list[dict[str, Any]]) -> str:
        if any(x in text for x in ["no me habla", "no quiere hablar", "encerró", "encerro", "callado", "shutdown", "se aisló", "se aislo"]):
            return "shutdown"
        if any(x in text for x in ["grita", "golpea", "explota", "meltdown", "se desbordó", "se desbordo", "berrinche", "berrinches", "se tira al piso"]):
            return "meltdown"
        if any(x in text for x in ["escuela", "maestra", "docente", "clase", "tarea", "salón", "salon", "alumno", "compañeros", "companeros"]):
            return "escuela_inclusiva"
        if any(x in text for x in ["duerme", "sueño", "sueno", "insomnio", "noche"]):
            return "sueno"
        if any(x in text for x in ["estres", "estrés", "agotada", "agotado", "rebasada", "rebasado", "cansada", "cansado", "ya no puedo"]):
            return "acompanamiento_cuidador"
        if any(x in text for x in ["acercarme", "empatizar", "conectar", "cariñosos", "carinosos", "afecto", "vínculo", "vinculo", "relación", "relacion", "nietos", "hijos", "amistad", "confianza", "cercanía", "cercania"]):
            return "vinculo_familiar"

        temas = [m.get("valor", "") for m in memory if m.get("categoria") == "tema_frecuente"]
        if temas:
            return Counter(temas).most_common(1)[0][0]
        return ctx.last_topic or "acompanamiento_general"

    def _detect_intent(self, text: str, ctx: SessionContext) -> str:
        if self._needs_clinical_boundary(text):
            return "consulta_clinica"
        if any(x in text for x in self.web_info_patterns):
            return "consulta_informativa"
        if any(x in text for x in ["cómo", "como", "qué hago", "que hago", "qué le digo", "que le digo", "qué puedo hacer", "que puedo hacer", "controlarlo", "controlarla"]):
            return "orientacion_practica"
        if any(x in text for x in ["por qué", "por que", "quisiera entender", "quiero entender", "no entiendo"]):
            return "comprension"
        if any(x in text for x in ["me duele", "me siento", "me pone triste", "me lastima", "me preocupa", "quisiera que se acercaran"]):
            return "acompanamiento_emocional"
        if text in ["sí", "si", "ok", "vale", "aja", "ajá", "bien"] or len(text.split()) <= 4:
            return "seguimiento" if ctx.case_active else "acompanamiento"
        return "acompanamiento"

    def _detect_emotion(self, text: str) -> str:
        if any(x in text for x in ["agotada", "agotado", "rebasada", "rebasado", "estres", "estrés", "cansada", "cansado", "ya no puedo"]):
            return "agotamiento"
        if any(x in text for x in ["preocupada", "preocupado", "miedo", "ansiedad", "ansiosa", "ansioso"]):
            return "ansiedad"
        if any(x in text for x in ["triste", "dolor", "me duele", "me lastima", "sola", "solo"]):
            return "tristeza"
        return "acompanamiento"

    def _detect_crisis(self, text: str, topic: str) -> str:
        if any(x in text for x in self.high_risk_patterns):
            return "riesgo_alto"
        if topic == "shutdown":
            return "shutdown"
        if topic == "meltdown":
            return "meltdown"
        return "sin_crisis"

    def _estimate_intensity(self, text: str) -> float:
        score = 0.30
        for token in ["urgente", "ya no puedo", "muy mal", "demasiado", "encerró", "encerro", "callado", "grita", "meltdown", "shutdown", "me duele", "me lastima", "berrinche", "berrinches"]:
            if token in text:
                score += 0.08
        return min(0.95, round(score, 2))

    def _estimate_confidence(self, text: str, topic: str, intent: str) -> float:
        score = 0.56 + (0.14 if topic != "acompanamiento_general" else 0) + (0.09 if intent != "acompanamiento" else 0)
        return min(0.95, round(score, 2))

    def _needs_clinical_boundary(self, text: str) -> bool:
        return any(k in text for k in self.boundary_patterns)

    def _select_protocol(self, topic: str, clinical: str) -> str:
        if clinical == "prohibida":
            return "protocolo_seguridad_clinica"
        mapping = {
            "shutdown": "protocolo_shutdown",
            "meltdown": "protocolo_meltdown",
            "sueno": "protocolo_sueno",
            "escuela_inclusiva": "protocolo_escuela_inclusiva",
            "acompanamiento_cuidador": "protocolo_cuidador_rebasado",
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
        strong_signals = [
            "grita", "golpea", "berrinche", "berrinches", "se tira al piso",
            "no me habla", "no quiere hablar", "compañeros", "alumno",
            "escuela", "tarea", "nietos", "hijos", "controlarlo", "controlarla",
            "hace", "molesta"
        ]
        if len(text.split()) >= 12:
            return True
        if any(k in text for k in strong_signals):
            return True
        if ctx.turns >= 1 and topic == ctx.last_topic:
            return True
        if intent == "orientacion_practica" and len(text.split()) >= 8:
            return True
        return False

    def _depth_by_role(self, role: str, text: str, ctx: SessionContext) -> str:
        if role == "docente":
            return "alto"
        if role == "cuidador(a)":
            return "alto" if len(text.split()) > 16 or ctx.turns >= 1 else "medio"
        if role == "abuelo(a)":
            return "medio"
        if role in ["madre", "padre"]:
            return "alto" if ctx.turns >= 1 else "medio"
        return "medio"

    def _format_steps(self, steps: list[str], depth: str) -> str:
        if depth == "bajo":
            selected = steps[:3]
        elif depth == "medio":
            selected = steps[:4]
        else:
            selected = steps
        return "\n".join(f"{i+1}. {step}" for i, step in enumerate(selected))

    def _role_phrase(self, role: str) -> str:
        mapping = {
            "madre": " como madre",
            "padre": " como padre",
            "abuelo(a)": " desde tu lugar de abuelo o abuela",
            "cuidador(a)": " desde tu papel de cuidado",
            "docente": " desde tu lugar docente",
            "adolescente": "",
            "adulto neurodivergente": "",
        }
        return mapping.get(role, "")

    def _topic_to_human(self, topic: str) -> str:
        mapping = {
            "vinculo_familiar": "vínculo familiar",
            "shutdown": "momentos de cierre o saturación",
            "meltdown": "momentos de desborde",
            "escuela_inclusiva": "escuela",
            "acompanamiento_cuidador": "cansancio del cuidador",
            "sueno": "sueño",
        }
        return mapping.get(topic, topic.replace("_", " "))

    def _empathetic_opening(self, text: str, role: str, display_name: str) -> str:
        name = f", {display_name}" if display_name else ""
        if any(x in text for x in ["me duele", "triste", "lastima", "sola", "solo"]):
            return f"Te leo{name}. Y sí, eso puede doler bastante."
        if any(x in text for x in ["no sé", "no se", "confund", "perdida", "perdido"]):
            return f"Entiendo{name}. A veces una situación así deja muchas preguntas juntas."
        if role == "abuelo(a)":
            return f"Te acompaño{name}. Lo que pasa con los nietos puede tocar fibras muy profundas."
        if role == "docente":
            return f"Entiendo{name}. En el aula, una situación así puede desgastar mucho."
        return f"Gracias por abrir esto{name}. Lo tomo con cuidado."

    def _update_context(self, ctx: SessionContext, text: str, topic: str, intent: str, enough_context: bool) -> SessionContext:
        details = list(ctx.details_known)
        if text and text not in details:
            details.append(text[:140])

        summary = f"tema={topic}; intent={intent}; detalles={len(details)}"
        emotional_tone = self._detect_emotion(text.lower())
        user_need = "vinculo" if topic == "vinculo_familiar" else intent
        style = "accion" if enough_context else "exploracion"

        return SessionContext(
            summary=summary,
            last_topic=topic,
            last_intent=intent,
            turns=ctx.turns + 1,
            user_goal=ctx.user_goal or ("orientación práctica" if intent == "orientacion_practica" else ""),
            case_active=True,
            details_known=details[-6:],
            emotional_tone=emotional_tone,
            last_user_need=user_need,
            last_response_style=style,
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

        for detonante in ["ruido", "fiesta", "escuela", "cansancio", "pantalla", "tarea", "teléfono", "telefono"]:
            if detonante in text:
                updates["items"].append(
                    {
                        "categoria": "detonante",
                        "clave": "detonante_reportado",
                        "valor": detonante,
                        "fuente": "conversacion",
                        "nivel_confianza": 0.7,
                    }
                )

        display_name = (user_profile.get("display_name") or "").strip()
        if display_name:
            updates["items"].append(
                {
                    "categoria": "preferencia_interaccion",
                    "clave": "nombre_preferido",
                    "valor": display_name,
                    "fuente": "perfil",
                    "nivel_confianza": 0.95,
                }
            )

        return updates


_ENGINE = None


def get_ng_engine() -> NeuroGuiaEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = NeuroGuiaEngine()
    return _ENGINE
