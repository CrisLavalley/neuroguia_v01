from __future__ import annotations

import uuid
from datetime import datetime

import pandas as pd
import streamlit as st
from sqlalchemy import text

from src.auth_helpers import require_login
from src.database import get_engine, init_db, read_sql_df
from src.orchestrator import SessionContext, get_ng_engine

st.set_page_config(
    page_title="NeuroGuía",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()
user = require_login() or {"is_logged_in": False, "email": "", "name": "Usuario"}
db_engine = get_engine()
ng_engine = get_ng_engine()

if "guest_id" not in st.session_state:
    st.session_state.guest_id = f"guest_{uuid.uuid4().hex[:12]}"

user_identifier = (
    user.get("email")
    if user.get("is_logged_in") and user.get("email")
    else st.session_state.guest_id
)

for key, default in {
    "session_id": str(uuid.uuid4()),
    "messages": [],
    "turn": 0,
    "conversation_context": SessionContext().__dict__,
    "last_result": None,
    "session_started": False,
}.items():
    st.session_state.setdefault(key, default)

SHOW_DEBUG_PANEL = bool(st.secrets.get("SHOW_DEBUG_PANEL", False))


def upsert_usuario(data: dict) -> None:
    with db_engine.begin() as conn:
        conn.execute(
            text(
                '''
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
                '''
            ),
            data,
        )


def load_profile() -> pd.DataFrame:
    return read_sql_df("SELECT * FROM usuarios WHERE id_usuario = :id", {"id": user_identifier})


def load_user_memory() -> list[dict]:
    df = read_sql_df(
        '''
        SELECT * FROM memoria_usuario
        WHERE id_usuario = :id AND activo = TRUE
        ORDER BY ultima_actualizacion DESC
        ''',
        {"id": user_identifier},
    )
    return df.to_dict("records") if not df.empty else []


def save_memory_updates(items: list[dict]) -> None:
    if not items:
        return
    now = datetime.utcnow().isoformat()
    with db_engine.begin() as conn:
        for item in items:
            conn.execute(
                text(
                    '''
                    INSERT INTO memoria_usuario (
                        id_memoria, id_usuario, categoria, clave, valor,
                        nivel_confianza, fuente, ultima_actualizacion, activo
                    )
                    VALUES (
                        :id_memoria, :id_usuario, :categoria, :clave, :valor,
                        :nivel_confianza, :fuente, :ultima_actualizacion, :activo
                    )
                    '''
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


def ensure_session_started(rol_usuario: str, estado_referencia: str) -> None:
    if st.session_state.session_started:
        return
    st.session_state.session_started = True
    with db_engine.begin() as conn:
        conn.execute(
            text(
                '''
                INSERT INTO sesiones (
                    id_sesion, id_usuario, email_usuario, rol_usuario,
                    estado_referencia, opened_at, status
                )
                VALUES (
                    :id_sesion, :id_usuario, :email_usuario, :rol_usuario,
                    :estado_referencia, :opened_at, :status
                )
                ON CONFLICT (id_sesion) DO NOTHING
                '''
            ),
            {
                "id_sesion": st.session_state.session_id,
                "id_usuario": user_identifier,
                "email_usuario": user.get("email", ""),
                "rol_usuario": rol_usuario,
                "estado_referencia": estado_referencia,
                "opened_at": datetime.utcnow().isoformat(),
                "status": "abierta",
            },
        )


def save_session_feedback(percepcion_cambio: str, utilidad_percibida: str) -> None:
    with db_engine.begin() as conn:
        conn.execute(
            text(
                '''
                UPDATE sesiones
                SET percepcion_cambio=:percepcion_cambio,
                    utilidad_percibida=:utilidad_percibida,
                    closed_at=:closed_at,
                    status='cerrada'
                WHERE id_sesion=:id_sesion
                '''
            ),
            {
                "percepcion_cambio": percepcion_cambio,
                "utilidad_percibida": utilidad_percibida,
                "closed_at": datetime.utcnow().isoformat(),
                "id_sesion": st.session_state.session_id,
            },
        )


def save_interaction(prompt: str, result, rol: str, estado: str) -> None:
    inter_id = str(uuid.uuid4())
    pd.DataFrame(
        [
            {
                "id_interaccion": inter_id,
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
            }
        ]
    ).to_sql("interacciones", db_engine, if_exists="append", index=False)


def start_new_session() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.turn = 0
    st.session_state.conversation_context = SessionContext().__dict__
    st.session_state.last_result = None
    st.session_state.session_started = False


def get_active_profile() -> dict:
    df = load_profile()
    return df.iloc[0].to_dict() if not df.empty else {}


def intensity_label(value: float) -> str:
    if value >= 0.75:
        return "Alta"
    if value >= 0.5:
        return "Media"
    return "Baja"


def topic_to_human(topic: str) -> str:
    mapping = {
        "vinculo_familiar": "Vínculo familiar",
        "shutdown": "Cierre o saturación",
        "meltdown": "Desborde",
        "escuela_inclusiva": "Escuela",
        "acompanamiento_cuidador": "Cansancio del cuidador",
        "sueno": "Sueño",
        "acompanamiento_general": "Acompañamiento general",
    }
    return mapping.get(topic, topic.replace("_", " ").title())


st.markdown(
    '''
    <style>
    .stApp { background: linear-gradient(180deg, #f5f7fc 0%, #eef2ff 100%); }
    .block-container { padding-top: 1rem; padding-bottom: 1.5rem; max-width: 1500px; }
    section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e7eaf3; }
    .brandbar {
        background: linear-gradient(90deg, #1f3b72 0%, #2b4f90 100%);
        color: white; padding: 18px 22px; border-radius: 18px; margin-bottom: 18px;
        box-shadow: 0 10px 30px rgba(31,59,114,0.18);
    }
    .brandtitle { font-size: 2rem; font-weight: 800; margin-bottom: 6px; }
    .brandsub { font-size: 1rem; opacity: 0.92; line-height: 1.45; }
    .metric-card {
        background: white; border: 1px solid #e5eaf5; border-radius: 16px; padding: 14px 16px;
        box-shadow: 0 8px 20px rgba(39, 53, 93, 0.05); margin-bottom: 10px; min-height: 84px;
    }
    .metric-label { font-size: 0.85rem; color: #667085; margin-bottom: 6px; }
    .metric-value { font-size: 1.35rem; font-weight: 800; color: #1f2937; }
    .panel-card {
        background: white; border: 1px solid #e5eaf5; border-radius: 18px; padding: 18px;
        box-shadow: 0 10px 24px rgba(39, 53, 93, 0.05); margin-bottom: 14px;
    }
    .panel-title { font-size: 1.1rem; font-weight: 800; color: #24324a; margin-bottom: 12px; }
    .status-box {
        background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 14px; padding: 14px; margin-bottom: 12px;
    }
    .status-label { font-size: 0.85rem; color: #667085; margin-bottom: 6px; }
    .status-value { font-size: 1rem; font-weight: 700; color: #111827; }
    .soft-badge {
        display:inline-block; padding: 5px 10px; border-radius: 999px; font-size: 0.82rem;
        background: #e8f0ff; color: #2859c5; font-weight: 700; margin-bottom: 10px;
    }
    .privacy-note {
        background: #f9fafb; border: 1px dashed #d0d5dd; border-radius: 12px; padding: 12px 14px;
        color: #475467; font-size: 0.92rem;
    }
    .stChatMessage { background: transparent !important; }
    .chat-shell {
        background: white; border: 1px solid #e5eaf5; border-radius: 18px; padding: 16px;
        box-shadow: 0 10px 24px rgba(39, 53, 93, 0.05);
    }
    .helper-box {
        background: #eef4ff; border: 1px solid #dce8ff; border-radius: 14px; padding: 12px 14px;
        color: #35507a; font-size: 0.95rem; margin-bottom: 12px;
    }
    </style>
    ''',
    unsafe_allow_html=True,
)

profile_df = load_profile()
perfil_existente = not profile_df.empty

with st.sidebar:
    st.markdown("## NeuroGuía")
    st.caption("Apoyo socioemocional no clínico")

    if perfil_existente:
        perfil = profile_df.iloc[0].to_dict()
        st.success(f"Perfil activo: {perfil.get('rol_usuario', 'usuario')}")
        with st.expander("Editar perfil", expanded=False):
            with st.form("form_editar_perfil"):
                opciones_rol = ["madre", "padre", "abuelo(a)", "cuidador(a)", "docente", "adolescente", "adulto neurodivergente", "otro"]
                rol_edit = st.selectbox(
                    "¿Cómo te identificas dentro de esta situación?",
                    opciones_rol,
                    index=opciones_rol.index(perfil.get("rol_usuario", "madre"))
                    if perfil.get("rol_usuario", "madre") in opciones_rol else 0,
                )
                nombre_edit = st.text_input("Nombre o cómo te gustaría que te llame NeuroGuía", perfil.get("nombre_mostrado", ""))
                estado_edit = st.text_input("Estado o región", perfil.get("estado_referencia", "Hidalgo"))
                red_edit = st.selectbox(
                    "¿Cuentas con red de apoyo?",
                    ["sí", "no", "parcialmente"],
                    index=["sí", "no", "parcialmente"].index(perfil.get("red_apoyo", "sí"))
                    if perfil.get("red_apoyo", "sí") in ["sí", "no", "parcialmente"] else 0,
                )
                consentimiento_edit = st.checkbox(
                    "Si el uso corresponde a una persona adolescente, confirmo acompañamiento adulto cuando aplique.",
                    value=bool(perfil.get("consentimiento_menor", False)),
                )
                guardar = st.form_submit_button("Guardar cambios")
            if guardar:
                now = datetime.utcnow().isoformat()
                upsert_usuario(
                    {
                        "id_usuario": user_identifier,
                        "email_usuario": user.get("email", ""),
                        "nombre_mostrado": (nombre_edit or "").strip(),
                        "rol_usuario": rol_edit,
                        "estado_referencia": estado_edit.strip() or "Hidalgo",
                        "red_apoyo": red_edit,
                        "consentimiento_menor": consentimiento_edit,
                        "created_at": perfil.get("created_at", now),
                        "updated_at": now,
                    }
                )
                st.success("Perfil actualizado.")
                st.rerun()
    else:
        st.info("Antes de comenzar, registra un perfil básico.")
        with st.form("form_onboarding"):
            rol_new = st.selectbox(
                "¿Cómo te identificas dentro de esta situación?",
                ["madre", "padre", "abuelo(a)", "cuidador(a)", "docente", "adolescente", "adulto neurodivergente", "otro"],
            )
            nombre_new = st.text_input("Nombre o cómo te gustaría que te llame NeuroGuía")
            estado_new = st.text_input("Estado o región", "Hidalgo")
            red_new = st.selectbox("¿Cuentas con red de apoyo?", ["sí", "no", "parcialmente"])
            consentimiento_new = st.checkbox("Si el uso corresponde a una persona adolescente, confirmo acompañamiento adulto cuando aplique.")
            guardar_new = st.form_submit_button("Guardar perfil y continuar")
        if guardar_new:
            now = datetime.utcnow().isoformat()
            upsert_usuario(
                {
                    "id_usuario": user_identifier,
                    "email_usuario": user.get("email", ""),
                    "nombre_mostrado": (nombre_new or "").strip(),
                    "rol_usuario": rol_new,
                    "estado_referencia": estado_new.strip() or "Hidalgo",
                    "red_apoyo": red_new,
                    "consentimiento_menor": consentimiento_new,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            st.success("Perfil guardado. Ya puedes comenzar.")
            st.rerun()
        st.stop()

perfil = get_active_profile()
rol = perfil.get("rol_usuario", "madre")
estado = perfil.get("estado_referencia", "Hidalgo")
red = perfil.get("red_apoyo", "sí")
display_name = (perfil.get("nombre_mostrado") or "").strip()
user_memory = load_user_memory()

with st.sidebar:
    with st.expander("Resumen", expanded=False):
        st.write(f"**Rol:** {rol}")
        if display_name:
            st.write(f"**Nombre:** {display_name}")
        st.write(f"**Estado:** {estado}")
        st.write(f"**Red de apoyo:** {red}")

    with st.expander("Cierre de sesión", expanded=False):
        percepcion = st.radio("Después de esta conversación, ¿te sientes?", ["mejor", "igual", "peor"], horizontal=True)
        utilidad = st.radio("¿Te fue útil esta conversación?", ["sí", "más o menos", "no"], horizontal=True)
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Guardar cierre"):
                ensure_session_started(rol, estado)
                save_session_feedback(percepcion, utilidad)
                st.success("Quedó guardado.")
        with col_b:
            if st.button("Nueva sesión"):
                start_new_session()
                st.rerun()

session_status = "Activa" if st.session_state.session_started else "Lista"
last_result = st.session_state.last_result
emotion = last_result.emocion if last_result else "Sin análisis"
topic = topic_to_human(last_result.topic) if last_result else "Sin tema"
source = last_result.fuente_respuesta if last_result else "Sin respuesta"
intensity = intensity_label(last_result.intensidad) if last_result else "Sin intensidad"

st.markdown(
    f'''
    <div class="brandbar">
        <div class="brandtitle">NeuroGuía</div>
        <div class="brandsub">Sistema híbrido de apoyo socioemocional no clínico para contextos de neurodivergencia. Diseñado para acompañar con calidez, claridad y orientación práctica.</div>
    </div>
    ''',
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Sesión</div><div class="metric-value">{session_status}</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Perfil activo</div><div class="metric-value">{rol}</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Tema actual</div><div class="metric-value">{topic}</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Fuente</div><div class="metric-value">{source}</div></div>', unsafe_allow_html=True)

left, right = st.columns([3.3, 1.4], gap="large")

with left:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="soft-badge">Chat de acompañamiento</div>', unsafe_allow_html=True)
    saludo = f"Hola, {display_name}" if display_name else "Hola"
    st.markdown(
        f'<div class="helper-box">{saludo}. Puedes escribir tal como hablas. Cuando ya haya suficiente contexto, NeuroGuía intentará responder con orientación concreta y no con preguntas repetitivas.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)

    if not st.session_state.messages:
        saludo_inicial = ng_engine.build_welcome_message(display_name=display_name, role=rol, user_memory=user_memory)
        st.session_state.messages.append(("assistant", saludo_inicial))

    for role_name, msg in st.session_state.messages:
        with st.chat_message(role_name):
            st.markdown(msg)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel-card"><div class="panel-title">Estado actual</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="status-box"><div class="status-label">Emoción detectada</div><div class="status-value">{emotion}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="status-box"><div class="status-label">Intensidad emocional</div><div class="status-value">{intensity}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="status-box"><div class="status-label">Tema detectado</div><div class="status-value">{topic}</div></div>',
        unsafe_allow_html=True,
    )
    strategy = last_result.protocolo if last_result else "Acompañamiento inicial"
    st.markdown(
        f'<div class="status-box"><div class="status-label">Estrategia aplicada</div><div class="status-value">{strategy}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="privacy-note"><strong>Privacidad y cuidado.</strong> Este espacio orienta y acompaña; no sustituye atención clínica ni evaluación profesional.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

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
            user_profile={
                "role": rol,
                "display_name": display_name,
                "state": estado,
                "support_network": red,
            },
            previous_messages=st.session_state.messages[:-1],
        )

        st.session_state.last_result = result
        st.session_state.conversation_context = result.context.__dict__
        st.session_state.messages.append(("assistant", result.respuesta))
        save_memory_updates(result.memory_updates.get("items", []))
        save_interaction(prompt, result, rol, estado)
        st.rerun()
    except Exception as exc:
        with st.chat_message("assistant"):
            st.error("Perdón, algo se atoró al procesar tu mensaje. Intenta de nuevo en unos segundos.")
        if SHOW_DEBUG_PANEL:
            st.exception(exc)

if SHOW_DEBUG_PANEL:
    with st.expander("Análisis del sistema", expanded=False):
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
