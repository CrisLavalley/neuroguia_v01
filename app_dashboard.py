
from __future__ import annotations

import streamlit as st

from src.database import init_db, read_sql_df

st.set_page_config(page_title="Dashboard NeuroGuía v10", page_icon="📊", layout="wide")
init_db()

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #fcfbff 0%, #f6f3ff 100%); }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Dashboard NeuroGuía")
st.caption("Seguimiento del prototipo: uso, interacciones, memoria y mejora progresiva.")

usuarios = read_sql_df("SELECT COUNT(*) AS total FROM usuarios")
sesiones = read_sql_df("SELECT COUNT(*) AS total FROM sesiones")
interacciones = read_sql_df("SELECT COUNT(*) AS total FROM interacciones")
memorias = read_sql_df("SELECT COUNT(*) AS total FROM memoria_usuario")
registros_mejora = read_sql_df("SELECT COUNT(*) AS total FROM analitica_mejora")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Usuarios", int(usuarios.iloc[0]['total']) if not usuarios.empty else 0)
c2.metric("Sesiones", int(sesiones.iloc[0]['total']) if not sesiones.empty else 0)
c3.metric("Interacciones", int(interacciones.iloc[0]['total']) if not interacciones.empty else 0)
c4.metric("Memorias activas", int(memorias.iloc[0]['total']) if not memorias.empty else 0)
c5.metric("Registros de mejora", int(registros_mejora.iloc[0]['total']) if not registros_mejora.empty else 0)

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
SELECT COALESCE(tema_detectado, 'sin dato') AS tema_detectado, COUNT(*) AS total
FROM analitica_mejora
GROUP BY COALESCE(tema_detectado, 'sin dato')
ORDER BY total DESC
""")

cierres = read_sql_df("""
SELECT COALESCE(percepcion_cambio, 'sin dato') AS percepcion_cambio, COUNT(*) AS total
FROM sesiones
GROUP BY COALESCE(percepcion_cambio, 'sin dato')
ORDER BY total DESC
""")

utilidad = read_sql_df("""
SELECT COALESCE(utilidad_percibida, 'sin dato') AS utilidad_percibida, COUNT(*) AS total
FROM sesiones
GROUP BY COALESCE(utilidad_percibida, 'sin dato')
ORDER BY total DESC
""")

historial = read_sql_df("""
SELECT created_at, rol_usuario, emocion_detectada, crisis_tipo, intent_detectado, topic_detectado, filtro_clinico
FROM interacciones
ORDER BY created_at DESC
LIMIT 40
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

c, d = st.columns(2)
with c:
    st.subheader("Temas detectados")
    if not temas.empty:
        st.bar_chart(temas.set_index("tema_detectado"))
    st.dataframe(temas, use_container_width=True, hide_index=True)

with d:
    st.subheader("Cierre y utilidad")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Percepción de cambio**")
        if not cierres.empty:
            st.bar_chart(cierres.set_index("percepcion_cambio"))
        st.dataframe(cierres, use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**Utilidad percibida**")
        if not utilidad.empty:
            st.bar_chart(utilidad.set_index("utilidad_percibida"))
        st.dataframe(utilidad, use_container_width=True, hide_index=True)

st.subheader("Historial reciente")
if not historial.empty:
    st.dataframe(historial, use_container_width=True, hide_index=True)
else:
    st.info("Todavía no hay interacciones registradas.")
