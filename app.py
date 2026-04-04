
from __future__ import annotations

import uuid
from datetime import datetime

import pandas as pd
import streamlit as st
from sqlalchemy import text

from src.auth_helpers import require_login
from src.database import get_engine, init_db, read_sql_df
from src.neuroguia_engine import SessionContext, get_ng_engine

st.set_page_config(
    page_title="NeuroGuía v10",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
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
    return read_sql_df(
        "SELECT * FROM usuarios WHERE id_usuario = :id",
        {"id": user_identifier},
    )


def load_user_memory() -> list[dict]:
    df = read_sql_df(
        """
        SELECT * FROM memoria_usuario
        WHERE id_usuario = :id AND activo = TRUE
        ORDER BY ultima_actualizacion DESC
        """,
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
            },
        )


def save_session_feedback(percepcion_cambio: str, utilidad_percibida: str) -> None:
    with db_engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE sesiones
                SET percepcion_cambio=:percepcion_cambio,
                    utilidad_percibida=:utilidad_percibida,
                    closed_at=:closed_at,
                    status='cerrada'
                WHERE id_sesion=:id_sesion
                """
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

    with db_engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO analitica_mejora (
                    id_registro, id_interaccion, tema_detectado,
                    intencion_detectada, respuesta_util, tipo_fallo,
                    hubo_seguimiento, hubo_cierre, percepcion_cambio,
                    utilidad_percibida, requiere_revision,
                    observaciones_revision, version_motor
                )
                VALUES (
                    :id_registro, :id_interaccion, :tema_detectado,
                    :intencion_detectada, :respuesta_util, :tipo_fallo,
                    :hubo_seguimiento, :hubo_cierre, :percepcion_cambio,
                    :utilidad_percibida, :requiere_revision,
                    :observaciones_revision, :version_motor
                )
                """
            ),
            {
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
                "version_motor": "v10",
            },
        )


def start_new_session() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.turn = 0
    st.session_state.conversation_context = SessionContext().__dict__
    st.session_state.last_result = None
    st.session_state.session_started = False


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #fcfbff 0%, #f7f4ff 100%);
    }
    .block-container {
        max-width: 900px;
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }
    .shell {
        background: rgba(255,255,255,0.82);
        border: 1px solid #ece7ff;
        border-radius: 26px;
        padding: 22px 22px 14px 22px;
        box-shadow: 0 14px 34px rgba(90, 65, 155, 0.06);
        margin-bottom: 14px;
    }
    .brand {
        display:inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background:#f1eaff;
        color:#6941c6;
        font-size:0.88rem;
        font-weight:700;
        margin-bottom:10px;
    }
    .title {
        font-size: 2.2rem;
        font-weight: 800;
        color:#2e2540;
        margin-bottom:4px;
    }
    .subtitle {
        color:#615b74;
        line-height:1.55;
        font-size:1rem;
        margin-bottom:10px;
    }
    .privacy {
        margin-top:10px;
        color:#726b86;
        font-size:0.92rem;
    }
    .stChatMessage {
        background: transparent !important;
    }
    section[data-testid="stSidebar"] {
        background: #fcfbff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

user_df = load_profile()
perfil_existente = not user_df.empty

with st.sidebar:
    st.markdown("### Perfil")
    if perfil_existente:
        perfil = user_df.iloc[0].to_dict()
        st.success(f"Activo: {perfil.get('rol_usuario', 'usuario')}")
        with st.expander("Editar perfil", expanded=False):
            with st.form("form_editar_perfil"):
                opciones_rol = [
                    "madre", "padre", "abuelo(a)", "cuidador(a)",
                    "docente", "adolescente", "adulto neurodivergente", "otro"
                ]
                rol_edit = st.selectbox(
                    "¿Cómo te identificas dentro de esta situación?",
                    opciones_rol,
                    index=opciones_rol.index(perfil.get("rol_usuario", "madre"))
                    if perfil.get("rol_usuario", "madre") in opciones_rol else 0,
                )
                nombre_edit = st.text_input(
                    "Nombre o cómo te gustaría que te llame NeuroGuía",
                    perfil.get("nombre_mostrado", ""),
                )
                estado_edit = st.text_input(
                    "Estado o región",
                    perfil.get("estado_referencia", "Hidalgo"),
                )
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
                guardar_edit = st.form_submit_button("Guardar cambios")
            if guardar_edit:
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
            consentimiento_new = st.checkbox(
                "Si el uso corresponde a una persona adolescente, confirmo acompañamiento adulto cuando aplique."
            )
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

perfil = load_profile().iloc[0].to_dict()
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
        percepcion = st.radio(
            "Después de esta conversación, ¿te sientes?",
            ["mejor", "igual", "peor"],
            horizontal=True,
        )
        utilidad = st.radio(
            "¿Te fue útil esta conversación?",
            ["sí", "más o menos", "no"],
            horizontal=True,
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Guardar"):
                ensure_session_started(rol, estado)
                save_session_feedback(percepcion, utilidad)
                st.success("Quedó guardado.")
        with col_b:
            if st.button("Nueva"):
                start_new_session()
                st.rerun()

name_fragment = f", {display_name}" if display_name else ""
st.markdown('<div class="shell">', unsafe_allow_html=True)
st.markdown('<div class="brand">NeuroGuía v10 · apoyo conversacional</div>', unsafe_allow_html=True)
st.markdown('<div class="title">NeuroGuía</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="subtitle">Hola{name_fragment}. Este espacio busca acompañarte con calidez, claridad y pasos posibles. '
    'Puedes escribir como hablas, sin seguir un formato especial.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="privacy"><strong>Privacidad y cuidado.</strong> Este espacio orienta y acompaña; no sustituye atención clínica ni evaluación profesional.</div>',
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

if not st.session_state.messages:
    saludo_inicial = ng_engine.build_welcome_message(
        display_name=display_name,
        role=rol,
        user_memory=user_memory,
    )
    st.session_state.messages.append(("assistant", saludo_inicial))

for role_name, msg in st.session_state.messages:
    with st.chat_message(role_name):
        st.markdown(msg)

prompt = st.chat_input(
    "Cuéntame qué está pasando. Ej. Me duele sentir que mis nietos ya no se acercan a mí."
)

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
        )

        st.session_state.last_result = result
        st.session_state.conversation_context = result.context.__dict__
        st.session_state.messages.append(("assistant", result.respuesta))
        save_memory_updates(result.memory_updates.get("items", []))
        save_interaction(prompt, result, rol, estado)
        st.rerun()

    except Exception as exc:
        with st.chat_message("assistant"):
            st.error(
                "Perdón, algo se atoró al procesar tu mensaje. Intenta de nuevo en unos segundos."
            )
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
            if result.recurso_rag:
                st.write("**Apoyos recuperados:**")
                for hit in result.recurso_rag:
                    st.write(f"• {hit['title']} ({hit['topic']})")
