from __future__ import annotations
import traceback
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st
from sqlalchemy import text

from src.auth_helpers import require_login
from src.database import get_engine, init_db, read_sql_df
from src.neuroguia_engine import SessionContext, get_ng_engine

st.set_page_config(page_title="NeuroGuía v08", page_icon="🧠", layout="wide")

PREMIUM_CSS = """
<style>
:root {
  --bg: #f7f6fb;
  --card: #ffffff;
  --accent: #7c3aed;
  --accent-soft: #ede9fe;
  --text: #1f2937;
  --muted: #6b7280;
  --success: #ecfdf5;
  --success-text: #047857;
  --warning: #fff7ed;
  --warning-text: #c2410c;
  --border: #e5e7eb;
}
.stApp {
  background: linear-gradient(180deg, #faf8ff 0%, #f7f6fb 100%);
  color: var(--text);
}
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #ffffff 0%, #f6f3ff 100%);
  border-right: 1px solid rgba(124,58,237,.10);
}
.hero-card {
  background: linear-gradient(135deg, rgba(124,58,237,.12), rgba(56,189,248,.08));
  border: 1px solid rgba(124,58,237,.10);
  border-radius: 22px;
  padding: 1.1rem 1.3rem;
  margin-bottom: 1rem;
  box-shadow: 0 12px 28px rgba(17, 24, 39, .06);
}
.hero-badge {
  display:inline-block;
  background:#fff;
  color: var(--accent);
  border:1px solid rgba(124,58,237,.18);
  border-radius:999px;
  padding:.28rem .7rem;
  font-size:.84rem;
  font-weight:700;
  margin-bottom:.55rem;
}
.hero-title {
  font-size: 2.35rem;
  font-weight: 800;
  letter-spacing: -.03em;
  margin: .2rem 0;
}
.hero-subtitle {
  color: var(--muted);
  font-size: 1rem;
  line-height: 1.55;
}
.metric-chip {
  background: rgba(255,255,255,.8);
  border: 1px solid rgba(124,58,237,.10);
  border-radius: 16px;
  padding: .8rem .95rem;
  min-height: 88px;
}
.metric-label {
  font-size: .83rem;
  color: var(--muted);
}
.metric-value {
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--text);
}
.note-card {
  background: #fff;
  border: 1px solid var(--border);
  border-left: 5px solid var(--accent);
  border-radius: 16px;
  padding: 1rem;
  margin-top: .75rem;
}
.small-muted { color: var(--muted); font-size: .9rem; }
.user-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 1rem;
  box-shadow: 0 8px 24px rgba(17,24,39,.04);
}
.footer-box {
  background: #fff;
  border: 1px dashed rgba(124,58,237,.25);
  border-radius: 16px;
  padding: .95rem 1rem;
  margin-top: .9rem;
}
div[data-testid="stChatMessage"] {
  border-radius: 22px;
  padding: .3rem .15rem;
}
div[data-testid="stChatMessageContent"] {
  background: rgba(255,255,255,.75);
  border: 1px solid rgba(229,231,235,.9);
  border-radius: 18px;
  padding: .8rem 1rem;
  box-shadow: 0 8px 20px rgba(17,24,39,.04);
}
div[data-testid="stChatMessage"]:has(span[data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"] {
  background: linear-gradient(180deg, #ffffff 0%, #fbfbff 100%);
  border-left: 4px solid rgba(239,68,68,.8);
}
div[data-testid="stChatMessage"]:has(span[data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stChatMessageContent"] {
  border-left: 4px solid rgba(245,158,11,.85);
}
.stButton > button {
  border-radius: 14px;
  border: 1px solid rgba(124,58,237,.15);
  box-shadow: 0 6px 16px rgba(17,24,39,.04);
}
[data-testid="stExpander"] {
  border: 1px solid var(--border);
  border-radius: 16px;
}
</style>
"""
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

DEBUG_UI = bool(st.secrets.get("SHOW_DEBUG_PANEL", False))

def show_friendly_error(title: str, exc: Exception) -> None:
    st.error(title)
    st.caption("No fue tu culpa. La app encontró un detalle técnico y ya quedó contenido para que no se rompa la experiencia.")
    if DEBUG_UI:
        with st.expander("Detalle técnico"):
            st.code(traceback.format_exc())

try:
    init_db()
    user = require_login() or {"is_logged_in": False, "email": "", "name": "Usuario"}
    db_engine = get_engine()
    ng_engine = get_ng_engine()
except Exception as exc:
    show_friendly_error("No se pudo iniciar NeuroGuía en este momento.", exc)
    st.stop()

if "guest_id" not in st.session_state:
    st.session_state.guest_id = f"guest_{uuid.uuid4().hex[:12]}"

user_identifier = user.get("email") if user.get("is_logged_in") and user.get("email") else st.session_state.guest_id

DEFAULTS = {
    "session_id": str(uuid.uuid4()),
    "messages": [],
    "turn": 0,
    "conversation_context": SessionContext().__dict__,
    "last_result": None,
    "session_started": False,
}
for key, default in DEFAULTS.items():
    st.session_state.setdefault(key, default)

def upsert_usuario(data: dict) -> None:
    with db_engine.begin() as conn:
        conn.execute(text("""INSERT INTO usuarios (
            id_usuario, email_usuario, nombre_mostrado, rol_usuario, estado_referencia,
            red_apoyo, consentimiento_menor, created_at, updated_at
        )
        VALUES (
            :id_usuario, :email_usuario, :nombre_mostrado, :rol_usuario, :estado_referencia,
            :red_apoyo, :consentimiento_menor, :created_at, :updated_at
        )
        ON CONFLICT (id_usuario) DO UPDATE SET
            email_usuario=EXCLUDED.email_usuario,
            nombre_mostrado=EXCLUDED.nombre_mostrado,
            rol_usuario=EXCLUDED.rol_usuario,
            estado_referencia=EXCLUDED.estado_referencia,
            red_apoyo=EXCLUDED.red_apoyo,
            consentimiento_menor=EXCLUDED.consentimiento_menor,
            updated_at=EXCLUDED.updated_at
        """), data)

def load_profile() -> pd.DataFrame:
    return read_sql_df("SELECT * FROM usuarios WHERE id_usuario = :id", {"id": user_identifier})

def load_user_memory() -> list[dict]:
    df = read_sql_df(
        "SELECT * FROM memoria_usuario WHERE id_usuario = :id AND activo = TRUE ORDER BY ultima_actualizacion DESC",
        {"id": user_identifier},
    )
    return df.to_dict("records") if not df.empty else []

def save_memory_updates(items: list[dict]) -> None:
    if not items:
        return
    now = datetime.utcnow().isoformat()
    with db_engine.begin() as conn:
        for item in items:
            conn.execute(text("""INSERT INTO memoria_usuario (
                id_memoria, id_usuario, categoria, clave, valor, nivel_confianza, fuente,
                ultima_actualizacion, activo
            )
            VALUES (
                :id_memoria, :id_usuario, :categoria, :clave, :valor, :nivel_confianza, :fuente,
                :ultima_actualizacion, :activo
            )"""), {
                "id_memoria": str(uuid.uuid4()),
                "id_usuario": user_identifier,
                "categoria": item.get("categoria", "general"),
                "clave": item.get("clave", "general"),
                "valor": item.get("valor", ""),
                "nivel_confianza": item.get("nivel_confianza", 0.5),
                "fuente": item.get("fuente", "conversacion"),
                "ultima_actualizacion": now,
                "activo": True,
            })

def ensure_session_started(rol_usuario: str, estado_referencia: str) -> None:
    if st.session_state.session_started:
        return
    st.session_state.session_started = True
    with db_engine.begin() as conn:
        conn.execute(text("""INSERT INTO sesiones (
            id_sesion, id_usuario, email_usuario, rol_usuario, estado_referencia, opened_at, status
        )
        VALUES (
            :id_sesion, :id_usuario, :email_usuario, :rol_usuario, :estado_referencia, :opened_at, :status
        )
        ON CONFLICT (id_sesion) DO NOTHING"""), {
            "id_sesion": st.session_state.session_id,
            "id_usuario": user_identifier,
            "email_usuario": user.get("email", ""),
            "rol_usuario": rol_usuario,
            "estado_referencia": estado_referencia,
            "opened_at": datetime.utcnow().isoformat(),
            "status": "abierta",
        })

def save_session_feedback(percepcion_cambio: str, utilidad_percibida: str) -> None:
    with db_engine.begin() as conn:
        conn.execute(text("""UPDATE sesiones
            SET percepcion_cambio=:percepcion_cambio,
                utilidad_percibida=:utilidad_percibida,
                closed_at=:closed_at,
                status='cerrada'
            WHERE id_sesion=:id_sesion"""), {
            "percepcion_cambio": percepcion_cambio,
            "utilidad_percibida": utilidad_percibida,
            "closed_at": datetime.utcnow().isoformat(),
            "id_sesion": st.session_state.session_id,
        })

def start_new_session() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.turn = 0
    st.session_state.conversation_context = SessionContext().__dict__
    st.session_state.last_result = None
    st.session_state.session_started = False

# Header premium
st.markdown("""
<div class="hero-card">
  <div class="hero-badge">NeuroGuía v08 · prototipo funcional</div>
  <div class="hero-title">NeuroGuía v08</div>
  <div class="hero-subtitle">
    Apoyo conversacional con memoria segura, orientación concreta y límites éticos.
    La experiencia busca sentirse cálida, clara y verdaderamente útil.
  </div>
</div>
""", unsafe_allow_html=True)

user_df = load_profile()
perfil_existente = not user_df.empty

with st.sidebar:
    st.markdown("<div class='user-card'>", unsafe_allow_html=True)
    st.subheader("Perfil de usuario")
    if perfil_existente:
        perfil = user_df.iloc[0].to_dict()
        st.success(f"Perfil activo: {perfil.get('rol_usuario', 'usuario')}")
        with st.expander("Editar perfil"):
            with st.form("form_editar_perfil"):
                opciones_rol = ["madre","padre","abuelo(a)","cuidador(a)","docente","adolescente","adulto neurodivergente","otro"]
                rol_edit = st.selectbox(
                    "¿Cómo te identificas dentro de esta situación?",
                    opciones_rol,
                    index=opciones_rol.index(perfil.get("rol_usuario","madre")) if perfil.get("rol_usuario","madre") in opciones_rol else 0
                )
                nombre_edit = st.text_input("Nombre o cómo te gustaría que te llame NeuroGuía (opcional)", perfil.get("nombre_mostrado",""))
                estado_edit = st.text_input("Estado o región", perfil.get("estado_referencia","Hidalgo"))
                red_edit = st.selectbox(
                    "¿Cuentas con red de apoyo?",
                    ["sí","no","parcialmente"],
                    index=["sí","no","parcialmente"].index(perfil.get("red_apoyo","sí")) if perfil.get("red_apoyo","sí") in ["sí","no","parcialmente"] else 0
                )
                consentimiento_edit = st.checkbox(
                    "Si el uso corresponde a una persona adolescente, confirmo que existe acompañamiento o resguardo adulto cuando aplique.",
                    value=bool(perfil.get("consentimiento_menor", False))
                )
                guardar = st.form_submit_button("Guardar cambios")
            if guardar:
                try:
                    now = datetime.utcnow().isoformat()
                    upsert_usuario({
                        "id_usuario": user_identifier,
                        "email_usuario": user.get("email",""),
                        "nombre_mostrado": (nombre_edit or "").strip(),
                        "rol_usuario": rol_edit,
                        "estado_referencia": estado_edit.strip() or "Hidalgo",
                        "red_apoyo": red_edit,
                        "consentimiento_menor": consentimiento_edit,
                        "created_at": perfil.get("created_at", now),
                        "updated_at": now,
                    })
                    st.success("Perfil actualizado.")
                    st.rerun()
                except Exception as exc:
                    show_friendly_error("No se pudo actualizar el perfil.", exc)
    else:
        st.info("Antes de comenzar, registra un perfil básico.")
        with st.form("form_onboarding"):
            rol_new = st.selectbox("¿Cómo te identificas dentro de esta situación?", ["madre","padre","abuelo(a)","cuidador(a)","docente","adolescente","adulto neurodivergente","otro"])
            nombre_new = st.text_input("Nombre o cómo te gustaría que te llame NeuroGuía (opcional)")
            estado_new = st.text_input("Estado o región", "Hidalgo")
            red_new = st.selectbox("¿Cuentas con red de apoyo?", ["sí","no","parcialmente"])
            consentimiento_new = st.checkbox("Si el uso corresponde a una persona adolescente, confirmo que existe acompañamiento o resguardo adulto cuando aplique.")
            guardar_new = st.form_submit_button("Guardar perfil y continuar")
        if guardar_new:
            try:
                now = datetime.utcnow().isoformat()
                upsert_usuario({
                    "id_usuario": user_identifier,
                    "email_usuario": user.get("email",""),
                    "nombre_mostrado": (nombre_new or "").strip(),
                    "rol_usuario": rol_new,
                    "estado_referencia": estado_new.strip() or "Hidalgo",
                    "red_apoyo": red_new,
                    "consentimiento_menor": consentimiento_new,
                    "created_at": now,
                    "updated_at": now,
                })
                st.success("Perfil guardado. Ya puedes iniciar la conversación.")
                st.rerun()
            except Exception as exc:
                show_friendly_error("No se pudo guardar el perfil.", exc)
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()
    st.markdown("</div>", unsafe_allow_html=True)

perfil = load_profile().iloc[0].to_dict()
rol = perfil.get("rol_usuario","madre")
estado = perfil.get("estado_referencia","Hidalgo")
red = perfil.get("red_apoyo","sí")
display_name = (perfil.get("nombre_mostrado") or "").strip()
user_memory = load_user_memory()

top1, top2, top3 = st.columns(3)
with top1:
    st.markdown(f"<div class='metric-chip'><div class='metric-label'>Perfil activo</div><div class='metric-value'>{rol}</div></div>", unsafe_allow_html=True)
with top2:
    st.markdown(f"<div class='metric-chip'><div class='metric-label'>Estado / región</div><div class='metric-value'>{estado}</div></div>", unsafe_allow_html=True)
with top3:
    st.markdown(f"<div class='metric-chip'><div class='metric-label'>Red de apoyo</div><div class='metric-value'>{red}</div></div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Resumen")
    st.write(f"• Rol: {rol}")
    if display_name:
        st.write(f"• Nombre: {display_name}")
    st.write(f"• Estado/región: {estado}")
    st.write(f"• Red de apoyo: {red}")
    if user_memory:
        temas = [m.get("valor","") for m in user_memory if m.get("categoria") == "tema_frecuente"]
        detonantes = [m.get("valor","") for m in user_memory if m.get("categoria") == "detonante"]
        if temas:
            st.write("• Temas frecuentes:", ", ".join(sorted(set(temas))[:3]))
        if detonantes:
            st.write("• Detonantes registrados:", ", ".join(sorted(set(detonantes))[:3]))

    st.markdown("---")
    st.subheader("Cierre breve de sesión")
    percepcion = st.radio("Después de esta conversación, ¿te sientes?", ["mejor","igual","peor"], horizontal=True)
    utilidad = st.radio("¿Te fue útil esta conversación?", ["sí","más o menos","no"], horizontal=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Guardar cierre", use_container_width=True):
            try:
                ensure_session_started(rol, estado)
                save_session_feedback(percepcion, utilidad)
                st.success("Cierre guardado.")
            except Exception as exc:
                show_friendly_error("No se pudo guardar el cierre de sesión.", exc)
    with c2:
        if st.button("Nueva sesión", use_container_width=True):
            start_new_session()
            st.rerun()

if rol == "adolescente":
    st.info("NeuroGuía está en modo adolescente: respuestas más claras, breves y con límites de seguridad reforzados.")

if not st.session_state.messages:
    saludo = (
        f"Hola{', ' + display_name if display_name else ''}. "
        "Estoy aquí para ayudarte a ordenar lo que pasa y darte pasos concretos, sin juzgarte."
    )
    with st.chat_message("assistant", avatar="🧠"):
        st.markdown(saludo)
        st.caption("Puedes contarme una situación concreta o pedirme orientación general sobre vínculo, escuela, regulación emocional o cansancio del cuidador.")

for role_name, msg in st.session_state.messages:
    avatar = "🧠" if role_name == "assistant" else "🧑"
    with st.chat_message(role_name, avatar=avatar):
        st.markdown(msg)

prompt = st.chat_input("Escribe lo que está pasando. Ej. No sé cómo acercarme a mi hijo sin que se sienta presionado.")
if prompt:
    try:
        ensure_session_started(rol, estado)
        st.session_state.turn += 1
        st.session_state.messages.append(("user", prompt))
        result = ng_engine.analyze(prompt, st.session_state.conversation_context, user_memory=user_memory)
        st.session_state.last_result = result
        st.session_state.conversation_context = result.context.__dict__
        st.session_state.messages.append(("assistant", result.respuesta))
        save_memory_updates(result.memory_updates.get("items", []))

        inter_id = str(uuid.uuid4())
        pd.DataFrame([{
            "id_interaccion": inter_id,
            "session_id": st.session_state.session_id,
            "id_usuario": user_identifier,
            "email_usuario": user.get("email",""),
            "rol_usuario": rol,
            "estado_referencia": estado,
            "turno": st.session_state.turn,
            "mensaje_usuario": prompt,
            "emocion_detectada": result.emocion,
            "probabilidad": result.probabilidad,
            "intensidad": result.intensidad,
            "crisis_tipo": result.crisis_tipo,
            "filtro_clinico": result.filtro_clinico,
            "protocolo": result.protocolo,
            "respuesta_ia": result.respuesta,
            "fuente_respuesta": result.fuente_respuesta,
            "intent_detectado": result.intent,
            "topic_detectado": result.topic,
            "created_at": datetime.utcnow().isoformat()
        }]).to_sql("interacciones", db_engine, if_exists="append", index=False)

        with db_engine.begin() as conn:
            conn.execute(text("""INSERT INTO analitica_mejora (
                id_registro, id_interaccion, tema_detectado, intencion_detectada, respuesta_util,
                tipo_fallo, hubo_seguimiento, hubo_cierre, percepcion_cambio, utilidad_percibida,
                requiere_revision, observaciones_revision, version_motor
            )
            VALUES (
                :id_registro, :id_interaccion, :tema_detectado, :intencion_detectada, :respuesta_util,
                :tipo_fallo, :hubo_seguimiento, :hubo_cierre, :percepcion_cambio, :utilidad_percibida,
                :requiere_revision, :observaciones_revision, :version_motor
            )"""), {
                "id_registro": str(uuid.uuid4()),
                "id_interaccion": inter_id,
                "tema_detectado": result.topic,
                "intencion_detectada": result.intent,
                "respuesta_util": None,
                "tipo_fallo": None,
                "hubo_seguimiento": result.intent == "seguimiento",
                "hubo_cierre": False,
                "percepcion_cambio": None,
                "utilidad_percibida": None,
                "requiere_revision": False,
                "observaciones_revision": None,
                "version_motor": "v08-premium",
            })
        st.rerun()
    except Exception as exc:
        show_friendly_error("No pude procesar tu mensaje como esperaba.", exc)

if DEBUG_UI:
    with st.expander("Ver análisis del sistema"):
        result = st.session_state.last_result
        if result is None:
            st.caption("Todavía no hay análisis en esta sesión.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Emoción", result.emocion)
            c2.metric("Crisis", result.crisis_tipo)
            c3.metric("Intención", result.intent)
            c4, c5, c6 = st.columns(3)
            c4.metric("Tema", result.topic)
            c5.metric("Intensidad", f"{result.intensidad:.2f}")
            c6.metric("Filtro clínico", result.filtro_clinico)
            st.write("**Protocolo activo:**", result.protocolo)
            resumen = (st.session_state.get("conversation_context") or {}).get("summary", "")
            if resumen:
                st.write("**Resumen de sesión:**", resumen)
            if result.recurso_rag:
                st.write("**Apoyos recuperados:**")
                for hit in result.recurso_rag:
                    st.write(f"• {hit['title']} ({hit['topic']})")
else:
    st.markdown("""
    <div class="footer-box">
      <strong>Privacidad y cuidado.</strong> Esta interfaz está pensada para acompañar, orientar y ordenar lo observado.
      No sustituye atención clínica ni evaluación profesional.
    </div>
    """, unsafe_allow_html=True)
