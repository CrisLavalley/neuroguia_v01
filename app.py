from __future__ import annotations
import uuid
from datetime import datetime
import pandas as pd
import streamlit as st
from sqlalchemy import text
from src.auth_helpers import require_login
from src.database import get_engine, init_db, read_sql_df
from src.neuroguia_engine import SessionContext, get_ng_engine

st.set_page_config(page_title="NeuroGuía v08", page_icon="🧠", layout="wide")
init_db()
user = require_login() or {"is_logged_in": False, "email": "", "name": "Usuario"}
db_engine = get_engine()
ng_engine = get_ng_engine()
if "guest_id" not in st.session_state:
    st.session_state.guest_id = f"guest_{uuid.uuid4().hex[:12]}"
user_identifier = user.get("email") if user.get("is_logged_in") and user.get("email") else st.session_state.guest_id
for key, default in {"session_id": str(uuid.uuid4()), "messages": [], "turn": 0, "conversation_context": SessionContext().__dict__, "last_result": None, "session_started": False}.items():
    st.session_state.setdefault(key, default)

def upsert_usuario(data: dict) -> None:
    with db_engine.begin() as conn:
        conn.execute(text("""INSERT INTO usuarios (id_usuario, email_usuario, nombre_mostrado, rol_usuario, estado_referencia, red_apoyo, consentimiento_menor, created_at, updated_at)
        VALUES (:id_usuario, :email_usuario, :nombre_mostrado, :rol_usuario, :estado_referencia, :red_apoyo, :consentimiento_menor, :created_at, :updated_at)
        ON CONFLICT (id_usuario) DO UPDATE SET email_usuario=EXCLUDED.email_usuario, nombre_mostrado=EXCLUDED.nombre_mostrado, rol_usuario=EXCLUDED.rol_usuario, estado_referencia=EXCLUDED.estado_referencia, red_apoyo=EXCLUDED.red_apoyo, consentimiento_menor=EXCLUDED.consentimiento_menor, updated_at=EXCLUDED.updated_at"""), data)

def load_profile() -> pd.DataFrame:
    return read_sql_df("SELECT * FROM usuarios WHERE id_usuario = :id", {"id": user_identifier})

def load_user_memory() -> list[dict]:
    df = read_sql_df("SELECT * FROM memoria_usuario WHERE id_usuario = :id AND activo = TRUE ORDER BY ultima_actualizacion DESC", {"id": user_identifier})
    return df.to_dict("records") if not df.empty else []

def save_memory_updates(items: list[dict]) -> None:
    if not items: return
    now = datetime.utcnow().isoformat()
    with db_engine.begin() as conn:
        for item in items:
            conn.execute(text("""INSERT INTO memoria_usuario (id_memoria, id_usuario, categoria, clave, valor, nivel_confianza, fuente, ultima_actualizacion, activo)
            VALUES (:id_memoria, :id_usuario, :categoria, :clave, :valor, :nivel_confianza, :fuente, :ultima_actualizacion, :activo)"""), {
                "id_memoria": str(uuid.uuid4()), "id_usuario": user_identifier, "categoria": item.get("categoria","general"),
                "clave": item.get("clave","general"), "valor": item.get("valor",""), "nivel_confianza": item.get("nivel_confianza",0.5),
                "fuente": item.get("fuente","conversacion"), "ultima_actualizacion": now, "activo": True
            })

def ensure_session_started(rol_usuario: str, estado_referencia: str) -> None:
    if st.session_state.session_started: return
    st.session_state.session_started = True
    with db_engine.begin() as conn:
        conn.execute(text("""INSERT INTO sesiones (id_sesion, id_usuario, email_usuario, rol_usuario, estado_referencia, opened_at, status)
        VALUES (:id_sesion, :id_usuario, :email_usuario, :rol_usuario, :estado_referencia, :opened_at, :status)
        ON CONFLICT (id_sesion) DO NOTHING"""), {
            "id_sesion": st.session_state.session_id, "id_usuario": user_identifier, "email_usuario": user.get("email",""),
            "rol_usuario": rol_usuario, "estado_referencia": estado_referencia, "opened_at": datetime.utcnow().isoformat(), "status": "abierta"
        })

def save_session_feedback(percepcion_cambio: str, utilidad_percibida: str) -> None:
    with db_engine.begin() as conn:
        conn.execute(text("""UPDATE sesiones SET percepcion_cambio=:percepcion_cambio, utilidad_percibida=:utilidad_percibida, closed_at=:closed_at, status='cerrada' WHERE id_sesion=:id_sesion"""), {
            "percepcion_cambio": percepcion_cambio, "utilidad_percibida": utilidad_percibida, "closed_at": datetime.utcnow().isoformat(), "id_sesion": st.session_state.session_id
        })

def start_new_session() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.turn = 0
    st.session_state.conversation_context = SessionContext().__dict__
    st.session_state.last_result = None
    st.session_state.session_started = False

st.title("NeuroGuía v08")
st.caption("Memoria segura por usuario, base curada y mejora periódica a partir de interacciones reales.")
user_df = load_profile()
perfil_existente = not user_df.empty

with st.sidebar:
    st.subheader("Perfil de usuario")
    if perfil_existente:
        perfil = user_df.iloc[0].to_dict()
        with st.expander("Editar perfil"):
            with st.form("form_editar_perfil"):
                opciones_rol = ["madre","padre","abuelo(a)","cuidador(a)","docente","adolescente","adulto neurodivergente","otro"]
                rol = st.selectbox("¿Cómo te identificas dentro de esta situación?", opciones_rol, index=opciones_rol.index(perfil.get("rol_usuario","madre")) if perfil.get("rol_usuario","madre") in opciones_rol else 0)
                nombre = st.text_input("Nombre o cómo te gustaría que te llame NeuroGuía (opcional)", perfil.get("nombre_mostrado",""))
                estado = st.text_input("Estado o región", perfil.get("estado_referencia","Hidalgo"))
                red = st.selectbox("¿Cuentas con red de apoyo?", ["sí","no","parcialmente"], index=["sí","no","parcialmente"].index(perfil.get("red_apoyo","sí")) if perfil.get("red_apoyo","sí") in ["sí","no","parcialmente"] else 0)
                consentimiento = st.checkbox("Si el uso corresponde a una persona adolescente, confirmo que existe acompañamiento o resguardo adulto cuando aplique.", value=bool(perfil.get("consentimiento_menor",False)))
                guardar = st.form_submit_button("Guardar cambios")
            if guardar:
                now = datetime.utcnow().isoformat()
                upsert_usuario({"id_usuario": user_identifier, "email_usuario": user.get("email",""), "nombre_mostrado": (nombre or "").strip(), "rol_usuario": rol, "estado_referencia": estado.strip() or "Hidalgo", "red_apoyo": red, "consentimiento_menor": consentimiento, "created_at": perfil.get("created_at", now), "updated_at": now})
                st.success("Perfil actualizado.")
                st.rerun()
    else:
        st.info("Antes de comenzar, registra un perfil básico.")
        with st.form("form_onboarding"):
            rol = st.selectbox("¿Cómo te identificas dentro de esta situación?", ["madre","padre","abuelo(a)","cuidador(a)","docente","adolescente","adulto neurodivergente","otro"])
            nombre = st.text_input("Nombre o cómo te gustaría que te llame NeuroGuía (opcional)")
            estado = st.text_input("Estado o región", "Hidalgo")
            red = st.selectbox("¿Cuentas con red de apoyo?", ["sí","no","parcialmente"])
            consentimiento = st.checkbox("Si el uso corresponde a una persona adolescente, confirmo que existe acompañamiento o resguardo adulto cuando aplique.")
            guardar = st.form_submit_button("Guardar perfil y continuar")
        if guardar:
            now = datetime.utcnow().isoformat()
            upsert_usuario({"id_usuario": user_identifier, "email_usuario": user.get("email",""), "nombre_mostrado": (nombre or "").strip(), "rol_usuario": rol, "estado_referencia": estado.strip() or "Hidalgo", "red_apoyo": red, "consentimiento_menor": consentimiento, "created_at": now, "updated_at": now})
            st.success("Perfil guardado. Ya puedes iniciar la conversación.")
            st.rerun()
        st.stop()

perfil = load_profile().iloc[0].to_dict()
rol = perfil.get("rol_usuario","madre")
estado = perfil.get("estado_referencia","Hidalgo")
red = perfil.get("red_apoyo","sí")
display_name = (perfil.get("nombre_mostrado") or "").strip()
user_memory = load_user_memory()

with st.sidebar:
    st.success(f"Perfil activo: {rol}")
    st.markdown("### Resumen")
    st.write(f"• Rol: {rol}")
    if display_name: st.write(f"• Nombre: {display_name}")
    st.write(f"• Estado/región: {estado}")
    st.write(f"• Red de apoyo: {red}")
    if user_memory:
        temas = [m.get("valor","") for m in user_memory if m.get("categoria") == "tema_frecuente"]
        detonantes = [m.get("valor","") for m in user_memory if m.get("categoria") == "detonante"]
        if temas: st.write("• Temas frecuentes:", ", ".join(sorted(set(temas))[:3]))
        if detonantes: st.write("• Detonantes registrados:", ", ".join(sorted(set(detonantes))[:3]))
    st.markdown("---")
    st.subheader("Cierre breve de sesión")
    percepcion = st.radio("Después de esta conversación, ¿te sientes?", ["mejor","igual","peor"], horizontal=True)
    utilidad = st.radio("¿Te fue útil esta conversación?", ["sí","más o menos","no"], horizontal=True)
    if st.button("Guardar cierre de sesión"):
        ensure_session_started(rol, estado)
        save_session_feedback(percepcion, utilidad)
        st.success("Cierre guardado.")
    if st.button("Empezar nueva sesión"):
        start_new_session()
        st.rerun()

if rol == "adolescente":
    st.info("NeuroGuía está en modo adolescente: respuestas más claras, breves y con límites de seguridad reforzados.")

if not st.session_state.messages:
    saludo = f"Hola{', ' + display_name if display_name else ''}. Cuéntame qué está pasando y te responderé con pasos concretos cuando necesites orientación."
    with st.chat_message("assistant"):
        st.markdown(saludo)

for role_name, msg in st.session_state.messages:
    with st.chat_message(role_name):
        st.markdown(msg)

prompt = st.chat_input("Escribe lo que está pasando. Ej. Mi hijo llegó saturado y no quiere hablar.")
if prompt:
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
        "id_interaccion": inter_id, "session_id": st.session_state.session_id, "id_usuario": user_identifier, "email_usuario": user.get("email",""), "rol_usuario": rol, "estado_referencia": estado, "turno": st.session_state.turn, "mensaje_usuario": prompt, "emocion_detectada": result.emocion, "probabilidad": result.probabilidad, "intensidad": result.intensidad, "crisis_tipo": result.crisis_tipo, "filtro_clinico": result.filtro_clinico, "protocolo": result.protocolo, "respuesta_ia": result.respuesta, "fuente_respuesta": result.fuente_respuesta, "intent_detectado": result.intent, "topic_detectado": result.topic, "created_at": datetime.utcnow().isoformat()
    }]).to_sql("interacciones", db_engine, if_exists="append", index=False)

    with db_engine.begin() as conn:
        conn.execute(text("""INSERT INTO analitica_mejora (id_registro, id_interaccion, tema_detectado, intencion_detectada, respuesta_util, tipo_fallo, hubo_seguimiento, hubo_cierre, percepcion_cambio, utilidad_percibida, requiere_revision, observaciones_revision, version_motor)
        VALUES (:id_registro, :id_interaccion, :tema_detectado, :intencion_detectada, :respuesta_util, :tipo_fallo, :hubo_seguimiento, :hubo_cierre, :percepcion_cambio, :utilidad_percibida, :requiere_revision, :observaciones_revision, :version_motor)"""), {
            "id_registro": str(uuid.uuid4()), "id_interaccion": inter_id, "tema_detectado": result.topic, "intencion_detectada": result.intent, "respuesta_util": None, "tipo_fallo": None, "hubo_seguimiento": result.intent == "seguimiento", "hubo_cierre": False, "percepcion_cambio": None, "utilidad_percibida": None, "requiere_revision": False, "observaciones_revision": None, "version_motor": "v08"
        })
    st.rerun()

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
        if resumen: st.write("**Resumen de sesión:**", resumen)
        if result.recurso_rag:
            st.write("**Apoyos recuperados:**")
            for hit in result.recurso_rag:
                st.write(f"• {hit['title']} ({hit['topic']})")
