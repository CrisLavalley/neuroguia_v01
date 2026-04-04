from __future__ import annotations

import streamlit as st

from src.database import init_db, read_sql_df

st.set_page_config(page_title="Dashboard NeuroGuía híbrido", page_icon="📊", layout="wide")
init_db()

st.title("Dashboard NeuroGuía")
st.caption("Seguimiento del prototipo híbrido: uso, interacciones, memoria y mejora progresiva.")

usuarios = read_sql_df("SELECT COUNT(*) AS total FROM usuarios")
sesiones = read_sql_df("SELECT COUNT(*) AS total FROM sesiones")
interacciones = read_sql_df("SELECT COUNT(*) AS total FROM interacciones")
memorias = read_sql_df("SELECT COUNT(*) AS total FROM memoria_usuario")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Usuarios", int(usuarios.iloc[0]["total"]) if not usuarios.empty else 0)
c2.metric("Sesiones", int(sesiones.iloc[0]["total"]) if not sesiones.empty else 0)
c3.metric("Interacciones", int(interacciones.iloc[0]["total"]) if not interacciones.empty else 0)
c4.metric("Memorias activas", int(memorias.iloc[0]["total"]) if not memorias.empty else 0)

roles = read_sql_df("""
SELECT COALESCE(rol_usuario, 'sin dato') AS rol_usuario, COUNT(*) AS total
FROM usuarios
GROUP BY COALESCE(rol_usuario, 'sin dato')
ORDER BY total DESC
""")

emociones = read_sql_df("""
SELECT COALESCE(emocion_detectada, 'sin dato') AS emocion_detectada, COUNT(*) AS total
FROM interacciones
GROUP BY COALESCE(emocion_detectada, 'sin dato')
ORDER BY total DESC
""")

temas = read_sql_df("""
SELECT COALESCE(topic_detectado, 'sin dato') AS topic_detectado, COUNT(*) AS total
FROM interacciones
GROUP BY COALESCE(topic_detectado, 'sin dato')
ORDER BY total DESC
""")

a, b = st.columns(2)
with a:
    st.subheader("Distribución por rol")
    if not roles.empty:
        st.bar_chart(roles.set_index("rol_usuario"))
    st.dataframe(roles, use_container_width=True, hide_index=True)

with b:
    st.subheader("Emociones detectadas")
    if not emociones.empty:
        st.bar_chart(emociones.set_index("emocion_detectada"))
    st.dataframe(emociones, use_container_width=True, hide_index=True)

st.subheader("Temas detectados")
if not temas.empty:
    st.bar_chart(temas.set_index("topic_detectado"))
st.dataframe(temas, use_container_width=True, hide_index=True)
