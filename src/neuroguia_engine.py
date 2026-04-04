from __future__ import annotations
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
        self.boundary_patterns = ["diagnóstico","diagnostico","medicación","medicacion","medicamento","receta","dosis","terapia específica","terapia especifica","pastilla","fármaco","farmaco"]

    def analyze(self, message: str, context_dict: dict[str, Any] | None = None, user_memory: list[dict[str, Any]] | None = None) -> AnalysisResult:
        ctx = SessionContext(**(context_dict or {}))
        text = (message or "").strip()
        lowered = text.lower()
        memory = user_memory or []
        topic = self._detect_topic(lowered, ctx, memory)
        intent = self._detect_intent(lowered, ctx)
        emotion = self._detect_emotion(lowered)
        crisis = self._detect_crisis(lowered, topic)
        intensity = self._estimate_intensity(lowered)
        confidence = self._estimate_confidence(lowered, topic, intent)
        clinical = "prohibida" if self._needs_clinical_boundary(lowered) else "permitida"
        response = self._build_response(lowered, topic, intent, crisis, clinical, memory)
        new_ctx = self._update_context(ctx, text, topic, intent)
        memory_updates = self._propose_memory_updates(lowered, topic)
        return AnalysisResult(
            emocion=emotion, probabilidad=confidence, intensidad=intensity, crisis_tipo=crisis,
            filtro_clinico=clinical, protocolo=self._select_protocol(topic, clinical), respuesta=response,
            fuente_respuesta="motor_v08_memoria_rag_mejora", intent=intent, topic=topic,
            recurso_rag=self._retrieve_rag(lowered, topic), context=new_ctx, memory_updates=memory_updates
        )

    def _detect_topic(self, text: str, ctx: SessionContext, memory: list[dict[str, Any]]) -> str:
        if any(x in text for x in ["no me habla","no quiere hablar","encerró","encerro","callado","shutdown","se aisló","se aislo"]): return "shutdown"
        if any(x in text for x in ["grita","golpea","explota","meltdown","se desbordó","se desbordo"]): return "meltdown"
        if any(x in text for x in ["escuela","maestra","docente","clase","tarea"]): return "escuela_inclusiva"
        if any(x in text for x in ["duerme","sueño","sueno","insomnio","noche"]): return "sueno"
        if any(x in text for x in ["estres","estrés","agotada","rebasada","cansada","ya no puedo"]): return "acompanamiento_cuidador"
        temas = [m.get("valor","") for m in memory if m.get("categoria") == "tema_frecuente"]
        if temas: return Counter(temas).most_common(1)[0][0]
        return ctx.last_topic or "acompanamiento_general"

    def _detect_intent(self, text: str, ctx: SessionContext) -> str:
        if self._needs_clinical_boundary(text): return "consulta_clinica"
        if any(x in text for x in ["qué hago","que hago","qué le digo","que le digo","cómo hago","como hago","cómo le hago","como le hago"]): return "orientacion_practica"
        if any(x in text for x in ["por qué","por que","quiero entender","necesito entender"]): return "comprension"
        if text in ["sí","si","ok","vale","aja","ajá","bien"] or len(text.split()) <= 3: return "seguimiento" if ctx.case_active else "acompanamiento"
        return "acompanamiento"

    def _detect_emotion(self, text: str) -> str:
        if any(x in text for x in ["agotada","rebasada","estres","estrés","cansada","ya no puedo"]): return "agotamiento"
        if any(x in text for x in ["preocupada","miedo","ansiedad","ansiosa"]): return "ansiedad"
        return "acompanamiento"

    def _detect_crisis(self, text: str, topic: str) -> str:
        if any(x in text for x in ["suicid","se quiere hacer daño","se quiere lastimar","arma","convuls","no respira"]): return "riesgo_alto"
        if topic == "shutdown": return "shutdown"
        if topic == "meltdown": return "meltdown"
        return "sin_crisis"

    def _estimate_intensity(self, text: str) -> float:
        score = 0.35
        for token in ["urgente","ya no puedo","muy mal","demasiado","encerró","encerro","callado","grita","meltdown","shutdown"]:
            if token in text: score += 0.08
        return min(0.95, round(score,2))

    def _estimate_confidence(self, text: str, topic: str, intent: str) -> float:
        score = 0.55 + (0.15 if topic != "acompanamiento_general" else 0) + (0.10 if intent != "acompanamiento" else 0)
        return min(0.95, round(score,2))

    def _needs_clinical_boundary(self, text: str) -> bool:
        return any(k in text for k in self.boundary_patterns)

    def _select_protocol(self, topic: str, clinical: str) -> str:
        if clinical == "prohibida": return "protocolo_seguridad_clinica"
        return {"shutdown":"protocolo_shutdown","meltdown":"protocolo_meltdown","sueno":"protocolo_sueno","escuela_inclusiva":"protocolo_escuela_inclusiva","acompanamiento_cuidador":"protocolo_cuidador_rebasado"}.get(topic, "protocolo_acompanamiento")

    def _retrieve_rag(self, text: str, topic: str) -> list[dict[str, Any]]:
        hits = []
        for doc in self.rag_docs:
            score = 3 if doc.get("topic") == topic else 0
            for kw in doc.get("keywords", []):
                if kw in text: score += 1
            if score > 0: hits.append((score, doc))
        hits.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in hits[:3]]

    def _build_response(self, text: str, topic: str, intent: str, crisis: str, clinical: str, memory: list[dict[str, Any]]) -> str:
        if clinical == "prohibida":
            return "No puedo diagnosticar, indicar medicación ni sustituir a profesionales de salud. Sí puedo ayudarte a organizar lo observado, identificar detonantes y preparar preguntas para una consulta segura."
        if crisis == "riesgo_alto":
            return "Lo que describes suena a una situación de alto riesgo. Lo más seguro es buscar apoyo presencial inmediato con un adulto responsable o un servicio de emergencia. Quédate con la persona y reduce riesgos cercanos mientras consigues ayuda."
        if intent == "seguimiento":
            return self._follow_up_response(topic)
        temas = [m.get("valor","") for m in memory if m.get("categoria") == "tema_frecuente"]
        support_note = f" La vez pasada aparecía mucho el tema de {Counter(temas).most_common(1)[0][0]}; si esto se parece, podemos usarlo a tu favor." if temas else ""
        if topic == "shutdown":
            return "Si tu hijo no quiere hablarte, lo más importante es no forzarlo en ese momento.\n\nTe sugiero esto:\n1. Dale un poco de espacio primero; puede estar saturado.\n2. Acércate con una frase simple y sin presión: \"No tienes que hablar ahorita. Solo quiero que sepas que estoy aquí.\"\n3. Evita preguntas insistentes como \"¿qué te pasa?\" o \"¿por qué estás así?\".\n4. Observa qué pasó antes: ruido, cansancio, conflicto, escuela o sobrecarga social.\n5. Retoma la conversación más tarde, cuando se vea más regulado.\n\nMuchas veces no es que no quiera hablar contigo, sino que en ese momento no puede." + support_note + "\n\nSi quieres, cuéntame qué pasó antes de que se encerrara y lo aterrizamos más."
        if topic == "meltdown":
            return "Si hubo un desborde fuerte, primero hay que bajar intensidad, no discutir.\n\nHaz esto:\n1. reduce estímulos,\n2. habla poco y en tono bajo,\n3. evita regaños o preguntas seguidas,\n4. prioriza seguridad física y emocional,\n5. deja la explicación para después.\n\nSi quieres, te ayudo a pensar qué hacer durante el momento y qué hacer después."
        if topic == "acompanamiento_cuidador":
            return "Si hoy te sientes muy rebasada, no intentes resolver todo al mismo tiempo.\n\nHaz solo esto por ahora:\n1. elige una sola prioridad para las próximas dos horas,\n2. deja fuera lo que pueda esperar,\n3. usa una frase breve contigo misma: \"Hoy no necesito hacerlo perfecto, solo hacerlo posible.\",\n4. si tienes red de apoyo, pide una ayuda concreta, aunque sea pequeña.\n\nSi quieres, dime qué es lo más urgente de hoy y lo ordenamos juntas."
        if topic == "escuela_inclusiva":
            return "Si esto tiene que ver con la escuela, conviene separar tres cosas: qué ocurrió, qué necesita el alumno y qué ajuste sí se puede pedir.\n\nPuedes empezar con algo como: \"Quiero entender qué pasó y pensar un ajuste concreto para que esto no escale.\"\n\nSi me dices qué pasó en clase, te ayudo a redactar el siguiente paso."
        if topic == "sueno":
            return "Si el problema es el sueño, primero conviene observar patrón antes de sacar conclusiones.\n\nRevisa: hora de dormir, qué pasó una hora antes y si hubo ruido, pantalla, ansiedad o sobrecarga.\n\nSi quieres, te ayudo a convertir eso en un registro breve."
        if intent == "orientacion_practica":
            return "Voy a responderte de forma concreta.\n\nDime qué pasó justo antes, qué hizo después y qué te preocupa más, y te digo qué conviene hacer ahora mismo."
        if intent == "comprension":
            return "Entiendo que quieres saber por qué está pasando esto. A veces la conducta no significa rechazo; puede ser cansancio, saturación, frustración o dificultad para procesar lo que vivió.\n\nSi me cuentas qué pasó antes y cómo estaba después, te ayudo a interpretarlo."
        return "Gracias por contarlo. Dime qué pasó justo antes de esta situación y te respondo con una orientación más concreta."

    def _follow_up_response(self, topic: str) -> str:
        if topic == "shutdown":
            return "Bien. Entonces avancemos sin forzarlo.\n\nPrueba esto hoy:\n1. no le exijas hablar de inmediato,\n2. déjale una puerta abierta con una frase corta: \"Cuando quieras, aquí estoy.\",\n3. si más tarde se regula un poco, haz una sola pregunta breve, no varias.\n\nSi quieres, te puedo dar exactamente qué frase usar cuando vuelvas a acercarte."
        if topic == "acompanamiento_cuidador":
            return "Bien, vamos a lo concreto.\n\nEscríbeme solo una de estas tres opciones y trabajamos sobre ella:\n1. qué me preocupa más\n2. qué sí depende de mí hoy\n3. qué puede esperar"
        return "Perfecto. Vamos a aterrizarlo. Cuéntame cuál es la parte más urgente y te respondo con una acción concreta."

    def _update_context(self, ctx: SessionContext, text: str, topic: str, intent: str) -> SessionContext:
        details = list(ctx.details_known)
        if text and text not in details: details.append(text[:140])
        summary = f"tema={topic}; intent={intent}; detalles={len(details)}"
        return SessionContext(summary=summary, last_topic=topic, last_intent=intent, turns=ctx.turns + 1, user_goal=ctx.user_goal or ("orientación práctica" if intent == "orientacion_practica" else ""), case_active=True, details_known=details[-6:])

    def _propose_memory_updates(self, text: str, topic: str) -> dict[str, Any]:
        updates = {"items": []}
        if topic != "acompanamiento_general":
            updates["items"].append({"categoria": "tema_frecuente", "clave": "tema_frecuente", "valor": topic, "fuente": "conversacion", "nivel_confianza": 0.8})
        for detonante in ["ruido","fiesta","escuela","cansancio","pantalla","tarea"]:
            if detonante in text:
                updates["items"].append({"categoria": "detonante", "clave": "detonante_reportado", "valor": detonante, "fuente": "conversacion", "nivel_confianza": 0.7})
        return updates

_ENGINE = None

def get_ng_engine() -> NeuroGuiaEngine:
    global _ENGINE
    if _ENGINE is None: _ENGINE = NeuroGuiaEngine()
    return _ENGINE
