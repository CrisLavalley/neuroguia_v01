from __future__ import annotations
import streamlit as st
from src.database import init_db, read_sql_df

st.set_page_config(page_title="Dashboard NeuroGuía v08", page_icon="📊", layout="wide")
init_db()
st.title("Dashboard de NeuroGuía v08")
st.caption("Panel de seguimiento para tesis: uso, memoria segura, base curada y mejora periódica.")

usuarios = read_sql_df("SELECT COUNT(*) AS total FROM usuarios")
sesiones = read_sql_df("SELECT COUNT(*) AS total FROM sesiones")
interacciones = read_sql_df("SELECT COUNT(*) AS total FROM interacciones")
memorias = read_sql_df("SELECT COUNT(*) AS total FROM memoria_usuario")
documentos = read_sql_df("SELECT COUNT(*) AS total FROM base_conocimiento")
registros_mejora = read_sql_df("SELECT COUNT(*) AS total FROM analitica_mejora")

c1, c2, c3 = st.columns(3)
c1.metric("Usuarios", int(usuarios.iloc[0]['total']) if not usuarios.empty else 0)
c2.metric("Sesiones", int(sesiones.iloc[0]['total']) if not sesiones.empty else 0)
c3.metric("Interacciones", int(interacciones.iloc[0]['total']) if not interacciones.empty else 0)
c4, c5, c6 = st.columns(3)
c4.metric("Memorias activas", int(memorias.iloc[0]['total']) if not memorias.empty else 0)
c5.metric("Documentos curados", int(documentos.iloc[0]['total']) if not documentos.empty else 0)
c6.metric("Registros de mejora", int(registros_mejora.iloc[0]['total']) if not registros_mejora.empty else 0)

roles = read_sql_df("SELECT COALESCE(rol_usuario, 'sin dato') AS rol_usuario, COUNT(*) AS total FROM usuarios GROUP BY COALESCE(rol_usuario, 'sin dato') ORDER BY total DESC")
emociones = read_sql_df("SELECT COALESCE(emocion_detectada, 'sin dato') AS emocion_detectada, COUNT(*) AS total FROM interacciones GROUP BY COALESCE(emocion_detectada, 'sin dato') ORDER BY total DESC")
crisis = read_sql_df("SELECT COALESCE(crisis_tipo, 'sin dato') AS crisis_tipo, COUNT(*) AS total FROM interacciones GROUP BY COALESCE(crisis_tipo, 'sin dato') ORDER BY total DESC")
intenciones = read_sql_df("SELECT COALESCE(intencion_detectada, 'sin dato') AS intencion_detectada, COUNT(*) AS total FROM analitica_mejora GROUP BY COALESCE(intencion_detectada, 'sin dato') ORDER BY total DESC")
temas = read_sql_df("SELECT COALESCE(tema_detectado, 'sin dato') AS tema_detectado, COUNT(*) AS total FROM analitica_mejora GROUP BY COALESCE(tema_detectado, 'sin dato') ORDER BY total DESC")
cierres = read_sql_df("SELECT COALESCE(percepcion_cambio, 'sin dato') AS percepcion_cambio, COUNT(*) AS total FROM sesiones GROUP BY COALESCE(percepcion_cambio, 'sin dato') ORDER BY total DESC")
utilidad = read_sql_df("SELECT COALESCE(utilidad_percibida, 'sin dato') AS utilidad_percibida, COUNT(*) AS total FROM sesiones GROUP BY COALESCE(utilidad_percibida, 'sin dato') ORDER BY total DESC")
memoria_cat = read_sql_df("SELECT COALESCE(categoria, 'sin dato') AS categoria, COUNT(*) AS total FROM memoria_usuario GROUP BY COALESCE(categoria, 'sin dato') ORDER BY total DESC")
incidentes = read_sql_df("SELECT COALESCE(categoria_incidente, 'sin dato') AS categoria_incidente, COUNT(*) AS total FROM incidentes_respuesta GROUP BY COALESCE(categoria_incidente, 'sin dato') ORDER BY total DESC")
historial = read_sql_df("SELECT created_at, rol_usuario, emocion_detectada, crisis_tipo, intent_detectado, topic_detectado, filtro_clinico FROM interacciones ORDER BY created_at DESC LIMIT 30")

a, b = st.columns(2)
with a:
    st.subheader("Distribución por rol")
    st.dataframe(roles, use_container_width=True, hide_index=True)
with b:
    st.subheader("Percepción de cambio")
    st.dataframe(cierres, use_container_width=True, hide_index=True)

c, d = st.columns(2)
with c:
    st.subheader("Utilidad percibida")
    st.dataframe(utilidad, use_container_width=True, hide_index=True)
with d:
    st.subheader("Emociones detectadas")
    st.dataframe(emociones, use_container_width=True, hide_index=True)

e, f = st.columns(2)
with e:
    st.subheader("Tipos de crisis")
    st.dataframe(crisis, use_container_width=True, hide_index=True)
with f:
    st.subheader("Intenciones detectadas")
    st.dataframe(intenciones, use_container_width=True, hide_index=True)

g, h = st.columns(2)
with g:
    st.subheader("Temas detectados")
    st.dataframe(temas, use_container_width=True, hide_index=True)
with h:
    st.subheader("Memoria por categoría")
    st.dataframe(memoria_cat, use_container_width=True, hide_index=True)

st.subheader("Incidentes de respuesta")
st.dataframe(incidentes, use_container_width=True, hide_index=True)

st.subheader("Historial reciente")
if not historial.empty:
    st.dataframe(historial, use_container_width=True, hide_index=True)
else:
    st.caption("Todavía no hay interacciones guardadas.")
