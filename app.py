from __future__ import annotations

import uuid
from datetime import datetime

import pandas as pd
import streamlit as st
from sqlalchemy import text

from src.auth_helpers import require_login
from src.database import get_engine, init_db, read_sql_df
from src.orchestrator import SessionContext, get_ng_engine

st.set_page_config(page_title="NeuroGuía", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")

init_db()
db_engine = get_engine()
ng_engine = get_ng_engine()
user = require_login() or {"is_logged_in": False, "email": "", "name": "Usuario"}

if "guest_id" not in st.session_state:
    st.session_state.guest_id = f"guest_{uuid.uuid4().hex[:12]}"
user_identifier = user.get("email") if user.get("is_logged_in") and user.get("email") else st.session_state.guest_id

for key, default in {
    "session_id": str(uuid.uuid4()),
    "messages": [],
    "turn": 0,
    "conversation_context": SessionContext().__dict__,
    "last_result": None,
    "session_started": False,
}.items():
    st.session_state.setdefault(key, default)


def upsert_usuario(data: dict) -> None:
    with db_engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO usuarios (
                    id_usuario, email_usuario, nombre_mostrado, rol_usuario,
                    estado_referencia, red_apoyo, consentimiento_menor,
                    created_at, updated_at
                )
                VALUES (
                    :id_usuario, :email_usuario, :nombre_mostrado, :rol_usuario,
                    :estado_referencia, :red_apoyo, :consentimiento_menor,
                    :created_at, :updated_at
                )
                ON CONFLICT (id_usuario) DO UPDATE SET
                    email_usuario=EXCLUDED.email_usuario,
                    nombre_mostrado=EXCLUDED.nombre_mostrado,
                    rol_usuario=EXCLUDED.rol_usuario,
                    estado_referencia=EXCLUDED.estado_referencia,
                    red_apoyo=EXCLUDED.red_apoyo,
                    consentimiento_menor=EXCLUDED.consentimiento_menor,
                    updated_at=EXCLUDED.updated_at
                """
            ),
            data,
        )


def load_profile() -> pd.DataFrame:
    return read_sql_df("SELECT * FROM usuarios WHERE id_usuario = :id", {"id": user_identifier})


def load_user_memory() -> list[dict]:
    df = read_sql_df("SELECT * FROM memoria_usuario WHERE id_usuario=:id AND activo = TRUE ORDER BY ultima_actualizacion DESC", {"id": user_identifier})
    return df.to_dict("records") if not df.empty else []


def save_memory_updates(items: list[dict]) -> None:
    if not items:
        return
    now = datetime.utcnow().isoformat()
    with db_engine.begin() as conn:
        for item in items:
            conn.execute(
                text(
                    """
                    INSERT INTO memoria_usuario (
                        id_memoria, id_usuario, categoria, clave, valor,
                        nivel_confianza, fuente, ultima_actualizacion, activo
                    )
                    VALUES (
                        :id_memoria, :id_usuario, :categoria, :clave, :valor,
                        :nivel_confianza, :fuente, :ultima_actualizacion, :activo
                    )
                    """
                ),
                {
                    "id_memoria": str(uuid.uuid4()),
                    "id_usuario": user_identifier,
                    "categoria": item.get("categoria", "general"),
                    "clave": item.get("clave", "general"),
                    "valor": item.get("valor", ""),
                    "nivel_confianza": item.get("nivel_confianza", 0.5),
                    "fuente": item.get("fuente", "conversacion"),
                    "ultima_actualizacion": now,
                    "activo": True,
                },
            )


def save_interaction(prompt: str, result, rol: str, estado: str) -> None:
    pd.DataFrame([{
        "id_interaccion": str(uuid.uuid4()),
        "session_id": st.session_state.session_id,
        "id_usuario": user_identifier,
        "email_usuario": user.get("email", ""),
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
        "created_at": datetime.utcnow().isoformat(),
    }]).to_sql("interacciones", db_engine, if_exists="append", index=False)


def ensure_session_started(rol_usuario: str, estado_referencia: str) -> None:
    if st.session_state.session_started:
        return
    st.session_state.session_started = True
    with db_engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO sesiones (
                    id_sesion, id_usuario, email_usuario, rol_usuario,
                    estado_referencia, opened_at, status
                )
                VALUES (
                    :id_sesion, :id_usuario, :email_usuario, :rol_usuario,
                    :estado_referencia, :opened_at, :status
                )
                ON CONFLICT (id_sesion) DO NOTHING
                """
            ),
            {
                "id_sesion": st.session_state.session_id,
                "id_usuario": user_identifier,
                "email_usuario": user.get("email", ""),
                "rol_usuario": rol_usuario,
                "estado_referencia": estado_referencia,
                "opened_at": datetime.utcnow().isoformat(),
                "status": "abierta",
            }
        )

st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%); }
.block-container { max-width: 980px; padding-top: 1.2rem; }
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e6e9f2; }
.hero { background: linear-gradient(100deg, #26457c 0%, #335da8 100%); color: white; padding: 24px 26px; border-radius: 24px; margin-bottom: 16px; box-shadow: 0 12px 30px rgba(27,60,120,0.18); }
.hero h1 { font-size: 2.15rem; margin: 0 0 6px 0; }
.hero p { margin: 0; opacity: 0.95; line-height: 1.5; }
.subtle-pill { display:inline-block; padding: 6px 12px; border-radius: 999px; background:#eef4ff; color:#2f67d8; font-weight:700; font-size:0.82rem; margin-bottom:12px; }
.chat-shell { background: white; border:1px solid #e6e9f2; border-radius: 22px; padding: 18px; box-shadow: 0 10px 24px rgba(45,55,75,0.05); }
.helper-note { background:#f5f7ff; border:1px solid #dde5ff; color:#4a5f8a; border-radius:14px; padding:12px 14px; margin-bottom:14px; }
.user-chip { display:inline-block; padding:8px 12px; border-radius: 999px; background:#e9fff3; color:#1e7d52; font-weight:700; font-size:0.85rem; margin-bottom:10px; }
.soft-disclaimer { color:#667085; font-size:0.9rem; margin-top:12px; }
.mode-line { color:#7b8190; font-size:0.8rem; margin-top:8px; }
.stChatMessage { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

profile_df = load_profile()
with st.sidebar:
    st.markdown("## NeuroGuía")
    st.caption("Apoyo socioemocional no clínico")
    if not profile_df.empty:
        perfil = profile_df.iloc[0].to_dict()
        st.markdown(f'<div class="user-chip">Perfil activo: {perfil.get("rol_usuario", "usuario")}</div>', unsafe_allow_html=True)
        with st.expander("Editar perfil", expanded=False):
            with st.form("perfil"):
                roles = ["madre", "padre", "docente", "cuidador(a)", "abuelo(a)", "otro"]
                rol = st.selectbox("Rol", roles, index=roles.index(perfil.get("rol_usuario", "madre")) if perfil.get("rol_usuario", "madre") in roles else 0)
                nombre = st.text_input("Nombre preferido", perfil.get("nombre_mostrado", ""))
                estado = st.text_input("Estado o región", perfil.get("estado_referencia", "Hidalgo"))
                red = st.selectbox("¿Cuentas con red de apoyo?", ["sí", "no", "parcialmente"], index=0)
                ok = st.form_submit_button("Guardar cambios")
            if ok:
                now = datetime.utcnow().isoformat()
                upsert_usuario({
                    "id_usuario": user_identifier,
                    "email_usuario": user.get("email", ""),
                    "nombre_mostrado": nombre.strip(),
                    "rol_usuario": rol,
                    "estado_referencia": estado.strip() or "Hidalgo",
                    "red_apoyo": red,
                    "consentimiento_menor": False,
                    "created_at": perfil.get("created_at", now),
                    "updated_at": now,
                })
                st.rerun()
        with st.expander("Resumen técnico", expanded=False):
            result = st.session_state.last_result
            if result:
                st.write(f"**Tema:** {result.topic}")
                st.write(f"**Fuente:** {result.fuente_respuesta}")
    else:
        with st.form("onboard"):
            rol = st.selectbox("¿Cómo te identificas en esta situación?", ["madre", "padre", "docente", "cuidador(a)", "abuelo(a)", "otro"])
            nombre = st.text_input("Nombre preferido")
            estado = st.text_input("Estado o región", "Hidalgo")
            red = st.selectbox("¿Cuentas con red de apoyo?", ["sí", "no", "parcialmente"])
            ok = st.form_submit_button("Guardar perfil")
        if ok:
            now = datetime.utcnow().isoformat()
            upsert_usuario({
                "id_usuario": user_identifier,
                "email_usuario": user.get("email", ""),
                "nombre_mostrado": nombre.strip(),
                "rol_usuario": rol,
                "estado_referencia": estado.strip() or "Hidalgo",
                "red_apoyo": red,
                "consentimiento_menor": False,
                "created_at": now,
                "updated_at": now,
            })
            st.rerun()
        st.stop()

perfil = load_profile().iloc[0].to_dict()
display_name = perfil.get("nombre_mostrado", "").strip()
rol = perfil.get("rol_usuario", "madre")
estado = perfil.get("estado_referencia", "Hidalgo")
user_memory = load_user_memory()

st.markdown("""
<div class="hero">
  <h1>NeuroGuía</h1>
  <p>Espacio de apoyo socioemocional no clínico para contextos de neurodivergencia. La experiencia del usuario prioriza calidez, claridad y orientación práctica.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
st.markdown('<div class="subtle-pill">Chat de acompañamiento</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="helper-note">Hola, {display_name or "gracias por estar aquí"}. Puedes escribir con libertad. NeuroGuía va a intentar ordenar lo que pasa y responderte con algo útil, sin que tengas que resumir lo que sientes.</div>',
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.session_state.messages.append(("assistant", ng_engine.build_welcome_message(display_name=display_name, role=rol, user_memory=user_memory)))

for role_name, msg in st.session_state.messages:
    with st.chat_message(role_name):
        st.markdown(msg)

prompt = st.chat_input("Escribe aquí lo que está pasando...")

if prompt:
    try:
        ensure_session_started(rol, estado)
        st.session_state.turn += 1
        st.session_state.messages.append(("user", prompt))
        result = ng_engine.analyze(
            message=prompt,
            context_dict=st.session_state.conversation_context,
            user_memory=user_memory,
            user_profile={"role": rol, "display_name": display_name, "state": estado, "support_network": perfil.get("red_apoyo", "sí")},
            previous_messages=st.session_state.messages[:-1],
        )
        st.session_state.last_result = result
        st.session_state.conversation_context = result.context.__dict__
        st.session_state.messages.append(("assistant", result.respuesta))
        save_memory_updates(result.memory_updates.get("items", []))
        save_interaction(prompt, result, rol, estado)
        st.rerun()
    except Exception:
        with st.chat_message("assistant"):
            st.error("Perdón, algo se atoró al procesar tu mensaje. Intenta de nuevo en unos segundos.")

st.markdown('<div class="soft-disclaimer"><strong>Privacidad y cuidado.</strong> Este espacio orienta y acompaña; no sustituye atención clínica ni evaluación profesional.</div>', unsafe_allow_html=True)
if st.session_state.last_result:
    st.markdown(f'<div class="mode-line">Modo actual: {st.session_state.last_result.fuente_respuesta}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
