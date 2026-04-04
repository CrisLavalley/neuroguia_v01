
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.web_helper import wikipedia_summary

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
        self.boundary_patterns = [
            "diagnóstico", "diagnostico", "medicación", "medicacion", "medicamento",
            "receta", "dosis", "terapia específica", "terapia especifica",
            "pastilla", "fármaco", "farmaco", "tratamiento", "medicar",
            "psiquiatra", "neurologo", "neurólogo"
        ]
        self.high_risk_patterns = [
            "suicid", "se quiere hacer daño", "se quiere lastimar", "arma",
            "convuls", "no respira", "me quiero morir", "quitarme la vida"
        ]
        self.web_info_patterns = [
            "qué es", "que es", "explícame", "explicame", "información sobre",
            "informacion sobre", "háblame de", "hablame de", "quién es", "quien es"
        ]

    # --------------------------
    # Entrada pública
    # --------------------------
    def build_welcome_message(
        self,
        display_name: str = "",
        role: str = "",
        user_memory: list[dict[str, Any]] | None = None,
    ) -> str:
        name_part = f", {display_name}" if display_name else ""
        role_text = self._role_phrase(role)
        memory = user_memory or []
        temas = [m.get("valor", "") for m in memory if m.get("categoria") == "tema_frecuente"]
        gentle_memory = ""
        if temas:
            frequent = Counter(temas).most_common(1)[0][0]
            gentle_memory = f" Recuerdo que antes apareció el tema de **{self._topic_to_human(frequent)}**."

        return (
            f"Hola{name_part}. Estoy aquí para acompañarte{role_text} y ayudarte a poner en palabras lo que estás viviendo."
            f"{gentle_memory}\n\n"
            "Puedes escribirme tal como te salga: una duda, una situación concreta, algo que te dolió hoy, o incluso una pregunta general. "
            "Yo voy a responderte de forma cercana, útil y sin empujarte a seguir un formato rígido."
        )

    def analyze(
        self,
        message: str,
        context_dict: dict[str, Any] | None = None,
        user_memory: list[dict[str, Any]] | None = None,
        user_profile: dict[str, Any] | None = None,
    ) -> AnalysisResult:
        ctx = SessionContext(**(context_dict or {}))
        text = (message or "").strip()
        lowered = text.lower()
        memory = user_memory or []
        profile = user_profile or {}

        topic = self._detect_topic(lowered, ctx, memory)
        intent = self._detect_intent(lowered, ctx)
        emotion = self._detect_emotion(lowered)
        crisis = self._detect_crisis(lowered, topic)
        intensity = self._estimate_intensity(lowered)
        confidence = self._estimate_confidence(lowered, topic, intent)
        clinical = "prohibida" if self._needs_clinical_boundary(lowered) else "permitida"
        rag_hits = self._retrieve_rag(lowered, topic)

        web_hit = None
        if self._should_use_web(lowered, clinical, crisis, intent):
            web_hit = wikipedia_summary(self._clean_web_query(text))

        response = self._build_response(
            text=lowered,
            original_text=text,
            topic=topic,
            intent=intent,
            crisis=crisis,
            clinical=clinical,
            memory=memory,
            user_profile=profile,
            web_hit=web_hit,
        )

        new_ctx = self._update_context(ctx, text, topic, intent)
        memory_updates = self._propose_memory_updates(lowered, topic, profile)

        source_name = "motor_v09_memoria_rag"
        if web_hit:
            source_name = "motor_v09_memoria_rag_web"

        combined_hits = list(rag_hits)
        if web_hit:
            combined_hits.append(
                {
                    "title": web_hit["title"],
                    "topic": "consulta_informativa",
                    "content": web_hit["summary"],
                    "source": web_hit["source"],
                    "url": web_hit["url"],
                }
            )

        return AnalysisResult(
            emocion=emotion,
            probabilidad=confidence,
            intensidad=intensity,
            crisis_tipo=crisis,
            filtro_clinico=clinical,
            protocolo=self._select_protocol(topic, clinical),
            respuesta=response,
            fuente_respuesta=source_name,
            intent=intent,
            topic=topic,
            recurso_rag=combined_hits[:3],
            context=new_ctx,
            memory_updates=memory_updates,
        )

    # --------------------------
    # Detección
    # --------------------------
    def _detect_topic(self, text: str, ctx: SessionContext, memory: list[dict[str, Any]]) -> str:
        if any(x in text for x in ["no me habla", "no quiere hablar", "encerró", "encerro", "callado", "shutdown", "se aisló", "se aislo"]):
            return "shutdown"
        if any(x in text for x in ["grita", "golpea", "explota", "meltdown", "se desbordó", "se desbordo"]):
            return "meltdown"
        if any(x in text for x in ["escuela", "maestra", "docente", "clase", "tarea", "salón", "salon"]):
            return "escuela_inclusiva"
        if any(x in text for x in ["duerme", "sueño", "sueno", "insomnio", "noche"]):
            return "sueno"
        if any(x in text for x in ["estres", "estrés", "agotada", "agotado", "rebasada", "rebasado", "cansada", "cansado", "ya no puedo"]):
            return "acompanamiento_cuidador"
        if any(x in text for x in ["acercarme", "empatizar", "conectar", "cariñosos", "carinosos", "afecto", "vínculo", "vinculo", "relación", "relacion", "nietos", "hijos", "amistad", "confianza"]):
            return "vinculo_familiar"
        if any(x in text for x in ["qué es", "que es", "explícame", "explicame", "información", "informacion", "háblame", "hablame"]):
            return "consulta_informativa"

        temas = [m.get("valor", "") for m in memory if m.get("categoria") == "tema_frecuente"]
        if temas:
            return Counter(temas).most_common(1)[0][0]

        return ctx.last_topic or "acompanamiento_general"

    def _detect_intent(self, text: str, ctx: SessionContext) -> str:
        if self._needs_clinical_boundary(text):
            return "consulta_clinica"

        if self._should_use_web(text, "permitida", "sin_crisis", "acompanamiento"):
            return "consulta_informativa"

        if any(x in text for x in ["cómo", "como", "qué hago", "que hago", "qué le digo", "que le digo", "qué puedo hacer", "que puedo hacer"]):
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
        for token in ["urgente", "ya no puedo", "muy mal", "demasiado", "encerró", "encerro", "callado", "grita", "meltdown", "shutdown", "me duele", "me lastima"]:
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

    def _should_use_web(self, text: str, clinical: str, crisis: str, intent: str) -> bool:
        if clinical == "prohibida" or crisis == "riesgo_alto":
            return False
        if intent == "consulta_informativa":
            return True
        return any(p in text for p in self.web_info_patterns)

    def _clean_web_query(self, text: str) -> str:
        return text.replace("¿", "").replace("?", "").strip()

    # --------------------------
    # Respuesta
    # --------------------------
    def _build_response(
        self,
        text: str,
        original_text: str,
        topic: str,
        intent: str,
        crisis: str,
        clinical: str,
        memory: list[dict[str, Any]],
        user_profile: dict[str, Any],
        web_hit: dict[str, Any] | None = None,
    ) -> str:
        display_name = (user_profile.get("display_name") or "").strip()
        role = user_profile.get("role", "")
        opening = self._empathetic_opening(text, role, display_name)

        if clinical == "prohibida":
            return (
                f"{opening}\n\n"
                "No puedo diagnosticar, indicar medicación ni sustituir a profesionales de salud. "
                "Sí puedo ayudarte a ordenar lo que has observado, identificar detonantes, preparar preguntas seguras "
                "o pensar cómo explicarlo con calma en una consulta."
            )

        if crisis == "riesgo_alto":
            return (
                f"{opening}\n\n"
                "Lo que describes suena a una situación de alto riesgo. En este momento lo más seguro es buscar apoyo presencial inmediato "
                "con una persona adulta responsable o un servicio de emergencia. Quédate cerca, reduce riesgos del entorno y prioriza ayuda directa."
            )

        if web_hit:
            return (
                f"{opening}\n\n"
                f"Te comparto una explicación breve sobre **{web_hit['title']}**:\n\n"
                f"{web_hit['summary']}\n\n"
                "Puedo ayudarte a aterrizar esta información a tu situación concreta, siempre cuidando mantenernos en una orientación no clínica."
            )

        if topic == "vinculo_familiar":
            return self._respond_vinculo_familiar(opening, text, role)

        if topic == "shutdown":
            return (
                f"{opening}\n\n"
                "Si en ese momento la otra persona se cierra o deja de hablar, suele ayudar más bajar la presión que insistir. "
                "Puedes probar con una presencia tranquila y una frase corta, por ejemplo: "
                ""No necesitas hablar ahorita. Solo quiero que sepas que aquí estoy."\n\n"
                "Si quieres, dime qué pasó justo antes y te ayudo a pensar cómo volver a acercarte sin forzar."
            )

        if topic == "meltdown":
            return (
                f"{opening}\n\n"
                "Cuando hay un desborde fuerte, conviene priorizar seguridad y poca estimulación. "
                "Más que explicar o corregir en ese instante, ayuda hablar poco, bajar el tono y dejar la explicación para después.\n\n"
                "Si quieres, puedo ayudarte a pensar qué hacer durante el momento y qué hacer ya cuando todo se calme."
            )

        if topic == "acompanamiento_cuidador":
            return (
                f"{opening}\n\n"
                "Hoy no necesitas resolver todo de una sola vez. A veces ayuda mucho elegir una sola prioridad para las próximas horas "
                "y soltar, aunque sea por un rato, lo demás.\n\n"
                "Si quieres, escríbeme qué es lo más pesado de hoy y lo ordenamos juntas o juntos paso por paso."
            )

        if topic == "escuela_inclusiva":
            return (
                f"{opening}\n\n"
                "Si esto tiene que ver con la escuela, suele servir separar tres cosas: qué pasó, qué necesidad apareció y qué ajuste concreto podría pedirse.\n\n"
                "Si me cuentas la situación con un poquito más de detalle, te ayudo a redactar un siguiente paso claro y respetuoso."
            )

        if topic == "sueno":
            return (
                f"{opening}\n\n"
                "Con el sueño suele ayudar más observar patrón que sacar conclusiones rápidas. "
                "A veces influye lo que ocurrió una hora antes, el cansancio acumulado, la ansiedad o la sobrecarga del día.\n\n"
                "Si quieres, te ayudo a convertirlo en un registro simple para entenderlo mejor."
            )

        if intent == "acompanamiento_emocional":
            return (
                f"{opening}\n\n"
                "Lo que sientes tiene sentido. No siempre es fácil llevar por dentro una mezcla de amor, preocupación, cansancio o tristeza "
                "cuando una relación importante no se siente como uno quisiera.\n\n"
                "No necesitas explicarlo de forma perfecta. Si quieres, puedo ayudarte a poner orden en lo que más te está pesando ahorita."
            )

        if intent == "orientacion_practica":
            return (
                f"{opening}\n\n"
                "Voy contigo a algo concreto. Cuéntame qué fue lo que pasó, qué te gustaría que ocurriera distinto y qué es lo que más te preocupa, "
                "y te respondo con pasos realistas."
            )

        if intent == "comprension":
            return (
                f"{opening}\n\n"
                "A veces lo que vemos como distancia, rechazo o desinterés también puede estar relacionado con hábitos, saturación, etapa de vida, "
                "dificultad para expresar afecto o costumbre de vincularse desde otro lugar.\n\n"
                "Si quieres, podemos mirar tu caso sin juzgarlo y entender qué podría estar pasando."
            )

        temas = [m.get("valor", "") for m in memory if m.get("categoria") == "tema_frecuente"]
        memory_note = ""
        if temas:
            tema = self._topic_to_human(Counter(temas).most_common(1)[0][0])
            memory_note = f" También noto que antes apareció el tema de **{tema}**, así que podemos darle continuidad si te sirve."

        return (
            f"{opening}\n\n"
            "Gracias por contármelo. Estoy aquí para ayudarte a aterrizar lo que estás viviendo sin meterte en un molde rígido."
            f"{memory_note}\n\n"
            "Si quieres, puedes contarme la situación tal cual ocurrió, o simplemente decirme qué te gustaría que cambiara."
        )

    def _respond_vinculo_familiar(self, opening: str, text: str, role: str) -> str:
        role_phrase = {
            "abuelo(a)": "con los nietos",
            "madre": "con tus hijos",
            "padre": "con tus hijos",
            "cuidador(a)": "con la persona que acompañas",
        }.get(role, "con esa persona")

        if any(x in text for x in ["que ellos se acercaran", "que se acercaran a mí", "que me buscaran", "cariñosos", "carinosos"]):
            return (
                f"{opening}\n\n"
                f"Es muy humano desear más cercanía {role_phrase}. A veces uno no está buscando algo enorme, sino pequeños gestos que hagan sentir vínculo.\n\n"
                "Lo primero es no tomar toda la distancia como falta de cariño. En muchas personas, especialmente cuando están absorbidas por sus rutinas, su edad o su forma de ser, "
                "el afecto no siempre sale de forma espontánea.\n\n"
                "Suele ayudar más abrir espacios pequeños y naturales que pedir cercanía directamente. Por ejemplo:\n"
                "1. empezar con conversaciones cortas y amables sobre algo que les interese,\n"
                "2. compartir un momento sencillo sin exigir que hablen mucho,\n"
                "3. hacer una invitación concreta y ligera, como tomar algo juntos, ver algo breve o acompañarte en una tarea pequeña,\n"
                "4. mostrar interés antes que reproche, para que no sientan presión.\n\n"
                "Si quieres, te ayudo a pensar una forma muy natural de acercarte a ellos según la edad que tienen."
            )

        if any(x in text for x in ["por qué no son cariñosos", "porque no son carinosos", "no hablan", "solo están en el teléfono", "solo estan en el telefono"]):
            return (
                f"{opening}\n\n"
                "Eso puede doler mucho, sobre todo cuando una quisiera sentir más calor en el vínculo. "
                "No siempre significa rechazo. A veces hay costumbre, timidez emocional, distracción, etapa de vida o una forma distinta de convivir.\n\n"
                "Antes de pensar que no quieren estar contigo, puede ayudar preguntarte: "
                "¿en qué momentos sí se relajan un poco?, ¿qué temas sí les interesan?, ¿cuándo están menos absorbidos por el teléfono o por su rutina?\n\n"
                "Si quieres, puedo ayudarte a encontrar maneras de acercarte sin que se sienta forzado ni doloroso."
            )

        return (
            f"{opening}\n\n"
            "Cuando lo que duele es la distancia emocional, suele funcionar mejor construir vínculo desde momentos pequeños que desde grandes conversaciones forzadas.\n\n"
            "Podemos pensar juntas o juntos en frases, gestos o actividades muy sencillas para abrir cercanía de una forma natural. "
            "Si quieres, dime la edad aproximada de ellos y cómo suele ser la convivencia cuando están contigo."
        )

    def _empathetic_opening(self, text: str, role: str, display_name: str) -> str:
        name = f", {display_name}" if display_name else ""
        if any(x in text for x in ["me duele", "triste", "lastima", "sola", "solo"]):
            return f"Te leo{name}. Y sí, eso puede doler bastante."
        if any(x in text for x in ["no sé", "no se", "confund", "perdida", "perdido"]):
            return f"Entiendo{name}. A veces una situación así deja muchas preguntas juntas."
        if role == "abuelo(a)":
            return f"Te acompaño{name}. Lo que pasa con los nietos puede tocar fibras muy profundas."
        return f"Gracias por abrir esto{name}. Lo tomo con cuidado."

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

    # --------------------------
    # Contexto y memoria
    # --------------------------
    def _update_context(self, ctx: SessionContext, text: str, topic: str, intent: str) -> SessionContext:
        details = list(ctx.details_known)
        if text and text not in details:
            details.append(text[:140])

        summary = f"tema={topic}; intent={intent}; detalles={len(details)}"
        emotional_tone = self._detect_emotion(text.lower())
        user_need = "vinculo" if topic == "vinculo_familiar" else intent

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
