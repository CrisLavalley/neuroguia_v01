from __future__ import annotations
import streamlit as st
from src.database import init_db, read_sql_df

st.set_page_config(page_title="NeuroGuía - Dashboard investigación", page_icon="📊", layout="wide")
init_db()
st.title("Dashboard de investigación")
st.caption("Vista separada para análisis, métricas y trazabilidad. No forma parte de la experiencia principal del usuario.")

inter = read_sql_df("SELECT COUNT(*) AS total FROM interacciones")
ses = read_sql_df("SELECT COUNT(*) AS total FROM sesiones")
usuarios = read_sql_df("SELECT COUNT(*) AS total FROM usuarios")
c1, c2, c3 = st.columns(3)
c1.metric("Interacciones", int(inter.iloc[0]['total']) if not inter.empty else 0)
c2.metric("Sesiones", int(ses.iloc[0]['total']) if not ses.empty else 0)
c3.metric("Usuarios", int(usuarios.iloc[0]['total']) if not usuarios.empty else 0)

temas = read_sql_df("SELECT COALESCE(topic_detectado,'sin dato') AS tema, COUNT(*) AS total FROM interacciones GROUP BY COALESCE(topic_detectado,'sin dato') ORDER BY total DESC")
fuentes = read_sql_df("SELECT COALESCE(fuente_respuesta,'sin dato') AS fuente, COUNT(*) AS total FROM interacciones GROUP BY COALESCE(fuente_respuesta,'sin dato') ORDER BY total DESC")
a,b = st.columns(2)
with a:
    st.subheader("Temas")
    if not temas.empty:
        st.bar_chart(temas.set_index("tema"))
        st.dataframe(temas, use_container_width=True, hide_index=True)
with b:
    st.subheader("Fuentes")
    if not fuentes.empty:
        st.bar_chart(fuentes.set_index("fuente"))
        st.dataframe(fuentes, use_container_width=True, hide_index=True)
