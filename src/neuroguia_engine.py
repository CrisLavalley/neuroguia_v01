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
        self.boundary_patterns = [
            "diagnóstico","diagnostico","medicación","medicacion","medicamento","receta","dosis",
            "terapia específica","terapia especifica","pastilla","fármaco","farmaco"
        ]

    def analyze(
        self,
        message: str,
        context_dict: dict[str, Any] | None = None,
        user_memory: list[dict[str, Any]] | None = None
    ) -> AnalysisResult:
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
        rag_hits = self._retrieve_rag(lowered, topic)
        response = self._build_response(lowered, topic, intent, crisis, clinical, memory, ctx)
        new_ctx = self._update_context(ctx, text, topic, intent)
        memory_updates = self._propose_memory_updates(lowered, topic, intent)

        return AnalysisResult(
            emocion=emotion,
            probabilidad=confidence,
            intensidad=intensity,
            crisis_tipo=crisis,
            filtro_clinico=clinical,
            protocolo=self._select_protocol(topic, clinical),
            respuesta=response,
            fuente_respuesta="motor_v08_premium_memoria_rag",
            intent=intent,
            topic=topic,
            recurso_rag=rag_hits,
            context=new_ctx,
            memory_updates=memory_updates,
        )

    def _detect_topic(self, text: str, ctx: SessionContext, memory: list[dict[str, Any]]) -> str:
        if any(x in text for x in ["no me habla","no quiere hablar","encerró","encerro","callado","shutdown","se aisló","se aislo"]):
            return "shutdown"
        if any(x in text for x in ["grita","golpea","explota","meltdown","se desbordó","se desbordo"]):
            return "meltdown"
        if any(x in text for x in ["escuela","maestra","docente","clase","tarea"]):
            return "escuela_inclusiva"
        if any(x in text for x in ["duerme","sueño","sueno","insomnio","noche"]):
            return "sueno"
        if any(x in text for x in ["estres","estrés","agotada","rebasada","cansada","ya no puedo"]):
            return "acompanamiento_cuidador"
        if any(x in text for x in ["acercarme","empatizar","amigos","vínculo","vinculo","conectar","relación","relacion","confianza","convivir"]):
            return "vinculo_familiar"
        temas = [m.get("valor","") for m in memory if m.get("categoria") == "tema_frecuente"]
        if temas:
            return Counter(temas).most_common(1)[0][0]
        return ctx.last_topic or "acompanamiento_general"

    def _detect_intent(self, text: str, ctx: SessionContext) -> str:
        if self._needs_clinical_boundary(text):
            return "consulta_clinica"
        if any(x in text for x in ["cómo acercarme","como acercarme","empatizar","conectar con","ser más amigos","ser mas amigos","ganar confianza"]):
            return "vinculo_practico"
        if any(x in text for x in ["qué hago","que hago","qué le digo","que le digo","cómo hago","como hago","cómo le hago","como le hago","oriéntame","orientame"]):
            return "orientacion_practica"
        if any(x in text for x in ["por qué","por que","quiero entender","necesito entender"]):
            return "comprension"
        if any(x in text for x in ["no pasó nada","no paso nada","en general","solo quiero","solo necesito orientación","solo necesito orientacion"]):
            return "orientacion_general"
        if text in ["sí","si","ok","vale","aja","ajá","bien"] or len(text.split()) <= 3:
            return "seguimiento" if ctx.case_active else "acompanamiento"
        return "acompanamiento"

    def _detect_emotion(self, text: str) -> str:
        if any(x in text for x in ["agotada","agotado","rebasada","rebasado","estres","estrés","cansada","cansado","ya no puedo"]):
            return "agotamiento"
        if any(x in text for x in ["preocupada","preocupado","miedo","ansiedad","ansiosa","ansioso","no sé como","no se como"]):
            return "ansiedad"
        if any(x in text for x in ["culpa","fallando","mala madre","mal padre"]):
            return "culpa"
        return "acompanamiento"

    def _detect_crisis(self, text: str, topic: str) -> str:
        if any(x in text for x in ["suicid","se quiere hacer daño","se quiere lastimar","arma","convuls","no respira"]):
            return "riesgo_alto"
        if topic == "shutdown":
            return "shutdown"
        if topic == "meltdown":
            return "meltdown"
        return "sin_crisis"

    def _estimate_intensity(self, text: str) -> float:
        score = 0.32
        for token in ["urgente","ya no puedo","muy mal","demasiado","encerró","encerro","callado","grita","meltdown","shutdown","no sé como","no se como"]:
            if token in text:
                score += 0.08
        return min(0.95, round(score, 2))

    def _estimate_confidence(self, text: str, topic: str, intent: str) -> float:
        score = 0.56 + (0.14 if topic != "acompanamiento_general" else 0) + (0.10 if intent not in ["acompanamiento", "seguimiento"] else 0)
        return min(0.95, round(score, 2))

    def _needs_clinical_boundary(self, text: str) -> bool:
        return any(k in text for k in self.boundary_patterns)

    def _select_protocol(self, topic: str, clinical: str) -> str:
        if clinical == "prohibida":
            return "protocolo_seguridad_clinica"
        return {
            "shutdown": "protocolo_shutdown",
            "meltdown": "protocolo_meltdown",
            "sueno": "protocolo_sueno",
            "escuela_inclusiva": "protocolo_escuela_inclusiva",
            "acompanamiento_cuidador": "protocolo_cuidador_rebasado",
            "vinculo_familiar": "protocolo_acompanamiento",
        }.get(topic, "protocolo_acompanamiento")

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

    def _build_response(
        self,
        text: str,
        topic: str,
        intent: str,
        crisis: str,
        clinical: str,
        memory: list[dict[str, Any]],
        ctx: SessionContext
    ) -> str:
        if clinical == "prohibida":
            return (
                "No puedo diagnosticar, indicar medicación ni sustituir a profesionales de salud. "
                "Sí puedo ayudarte a organizar lo observado, identificar detonantes y preparar preguntas para una consulta segura."
            )

        if crisis == "riesgo_alto":
            return (
                "Lo que describes suena a una situación de alto riesgo. Lo más seguro es buscar apoyo presencial inmediato "
                "con un adulto responsable o un servicio de emergencia. Quédate con la persona y reduce riesgos cercanos mientras consigues ayuda."
            )

        if intent == "seguimiento":
            return self._follow_up_response(topic)

        temas = [m.get("valor","") for m in memory if m.get("categoria") == "tema_frecuente"]
        support_note = ""
        if temas:
            support_note = f"\n\nVeo que antes apareció el tema de **{Counter(temas).most_common(1)[0][0]}**; eso puede ayudarnos a ver patrones sin empezar de cero."

        if topic == "vinculo_familiar" or intent in ["vinculo_practico", "orientacion_general"]:
            return (
                "Gracias por decirlo tan claro. Lo que estás buscando no es una solución rápida, sino una forma más humana de acercarte a tus hijos, "
                "y eso ya habla muy bien de ti.\n\n"
                "Para empezar, no intentes conectar corrigiendo o interrogando. Intenta **acercarte desde lo compartido y lo seguro**:\n"
                "1. elige un momento sin presión, no cuando haya conflicto,\n"
                "2. acércate a algo que a ellos sí les interese,\n"
                "3. haz comentarios breves en vez de muchas preguntas,\n"
                "4. valida antes de aconsejar: *\"No quiero presionarte, solo quiero entenderte mejor\"*,\n"
                "5. busca pequeños rituales: una caminata corta, ver algo juntos, un juego, una bebida, cinco minutos de charla.\n\n"
                "Frases que sí ayudan:\n"
                "- *\"No necesito que me cuentes todo hoy. Solo quiero estar más cerca de ti.\"*\n"
                "- *\"Quiero aprender a acompañarte mejor, no controlarte más.\"*\n"
                "- *\"Si te cuesta hablar, podemos empezar por algo pequeño.\"*\n\n"
                "Lo importante no es volverse 'mejores amigos' de golpe, sino construir **confianza repetida en momentos pequeños**." +
                support_note +
                "\n\nSi quieres, te puedo dar ahora mismo un plan de 7 días para empezar a acercarte a ellos sin que se sienta forzado."
            )

        if topic == "shutdown":
            return (
                "Si tu hijo no quiere hablarte, lo más importante es no forzarlo en ese momento.\n\n"
                "Te sugiero esto:\n"
                "1. dale un poco de espacio primero; puede estar saturado,\n"
                "2. acércate con una frase simple y sin presión: *\"No tienes que hablar ahorita. Solo quiero que sepas que estoy aquí\"*,\n"
                "3. evita preguntas insistentes como *\"¿qué te pasa?\"*,\n"
                "4. observa qué ocurrió antes: ruido, cansancio, escuela, conflicto o sobrecarga social,\n"
                "5. retoma más tarde, cuando ya se vea más regulado.\n\n"
                "Muchas veces no es rechazo: es saturación." +
                support_note +
                "\n\nSi quieres, dime la edad de tu hijo y te doy frases más adecuadas para acercarte."
            )

        if topic == "meltdown":
            return (
                "Si hubo un desborde fuerte, primero hay que bajar intensidad, no discutir.\n\n"
                "Haz esto:\n"
                "1. reduce estímulos,\n"
                "2. habla poco y en tono bajo,\n"
                "3. evita regaños o preguntas seguidas,\n"
                "4. prioriza seguridad física y emocional,\n"
                "5. deja la explicación para después.\n\n"
                "Cuando el momento pase, te puedo ayudar a pensar qué detonó el desborde y cómo prevenir el siguiente."
            )

        if topic == "acompanamiento_cuidador":
            return (
                "Se siente pesado, y no tienes que cargarlo todo perfecto para estar haciéndolo bien.\n\n"
                "Por ahora prueba esto:\n"
                "1. elige una sola prioridad para las próximas dos horas,\n"
                "2. suelta lo que pueda esperar,\n"
                "3. usa una frase breve contigo: *\"Hoy no necesito hacerlo perfecto, solo hacerlo posible\"*,\n"
                "4. si tienes red de apoyo, pide una ayuda concreta, no general.\n\n"
                "Si quieres, dime qué te está rebasando más hoy y lo ordenamos juntas(os)."
            )

        if topic == "escuela_inclusiva":
            return (
                "Si esto tiene que ver con la escuela, conviene separar tres cosas: **qué ocurrió, qué necesita el alumno y qué ajuste sí se puede pedir**.\n\n"
                "Puedes empezar con algo como: *\"Quiero entender qué pasó y pensar un ajuste concreto para que esto no escale.\"*\n\n"
                "Si me dices qué pasó en clase, te ayudo a redactar el mensaje o la conversación."
            )

        if topic == "sueno":
            return (
                "Si el problema es el sueño, antes de sacar conclusiones conviene observar el patrón.\n\n"
                "Revisa: hora de dormir, qué ocurrió una hora antes y si hubo ruido, pantalla, ansiedad o sobrecarga.\n\n"
                "Si quieres, te ayudo a convertir eso en un registro breve y útil."
            )

        if intent == "orientacion_practica":
            return (
                "Voy contigo a lo concreto.\n\n"
                "Cuéntame estas tres cosas y te respondo con un paso siguiente bien aterrizado:\n"
                "1. qué pasó justo antes,\n"
                "2. qué hizo después,\n"
                "3. qué te preocupa más a ti en este momento."
            )

        if intent == "comprension":
            return (
                "Entiendo que quieres comprenderlo, no solo apagar la situación. A veces la conducta no significa rechazo; puede ser cansancio, saturación, frustración o dificultad para procesar lo vivido.\n\n"
                "Si me cuentas qué pasó antes y cómo estaba después, te ayudo a interpretarlo con más claridad."
            )

        return (
            "Gracias por contarlo. Para orientarte mejor necesito ubicar un poco el contexto, pero no quiero hacerte sentir en un ciclo.\n\n"
            "Puedes elegir una de estas rutas y te respondo directo:\n"
            "- **vínculo**: cómo acercarme sin presionar\n"
            "- **escuela**: qué hacer con una situación escolar\n"
            "- **desborde**: qué hacer durante un momento difícil\n"
            "- **cansancio**: cómo ordenar lo que me está rebasando"
        )

    def _follow_up_response(self, topic: str) -> str:
        if topic == "shutdown":
            return (
                "Bien. Entonces avancemos sin forzarlo.\n\n"
                "Prueba esto hoy:\n"
                "1. no le exijas hablar de inmediato,\n"
                "2. déjale una puerta abierta con una frase corta: *\"Cuando quieras, aquí estoy\"*,\n"
                "3. si más tarde se regula un poco, haz una sola pregunta breve.\n\n"
                "Si quieres, te doy exactamente qué frase usar."
            )
        if topic == "vinculo_familiar":
            return (
                "Perfecto. Entonces vamos por cercanía sin presión.\n\n"
                "Hoy intenta solo una de estas acciones:\n"
                "1. compartir 10 minutos de algo que a tu hijo sí le guste,\n"
                "2. hacer una pregunta ligera sin buscar una gran conversación,\n"
                "3. validar algo que le cueste sin corregirlo enseguida.\n\n"
                "Si me dices la edad de tus hijos, te doy ejemplos más precisos."
            )
        if topic == "acompanamiento_cuidador":
            return (
                "Bien, vamos a lo concreto.\n\n"
                "Escríbeme solo una de estas tres opciones y trabajamos sobre ella:\n"
                "1. qué me preocupa más\n"
                "2. qué sí depende de mí hoy\n"
                "3. qué puede esperar"
            )
        return "Perfecto. Vamos a aterrizarlo. Cuéntame cuál es la parte más urgente y te respondo con una acción concreta."

    def _update_context(self, ctx: SessionContext, text: str, topic: str, intent: str) -> SessionContext:
        details = list(ctx.details_known)
        if text and text not in details:
            details.append(text[:180])
        summary = f"tema={topic}; intent={intent}; detalles={len(details)}"
        return SessionContext(
            summary=summary,
            last_topic=topic,
            last_intent=intent,
            turns=ctx.turns + 1,
            user_goal=ctx.user_goal or ("orientación práctica" if intent in ["orientacion_practica", "vinculo_practico"] else ""),
            case_active=True,
            details_known=details[-6:],
        )

    def _propose_memory_updates(self, text: str, topic: str, intent: str) -> dict[str, Any]:
        updates = {"items": []}
        if topic != "acompanamiento_general":
            updates["items"].append({
                "categoria": "tema_frecuente",
                "clave": "tema_frecuente",
                "valor": topic,
                "fuente": "conversacion",
                "nivel_confianza": 0.8,
            })
        if intent in ["vinculo_practico", "orientacion_general"]:
            updates["items"].append({
                "categoria": "objetivo_usuario",
                "clave": "objetivo_usuario",
                "valor": "mejorar_vinculo_familiar",
                "fuente": "conversacion",
                "nivel_confianza": 0.8,
            })
        for detonante in ["ruido","fiesta","escuela","cansancio","pantalla","tarea"]:
            if detonante in text:
                updates["items"].append({
                    "categoria": "detonante",
                    "clave": "detonante_reportado",
                    "valor": detonante,
                    "fuente": "conversacion",
                    "nivel_confianza": 0.7,
                })
        return updates

_ENGINE = None

def get_ng_engine() -> NeuroGuiaEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = NeuroGuiaEngine()
    return _ENGINE
