from __future__ import annotations
import altair as alt
import pandas as pd
import streamlit as st

from src.database import init_db, read_sql_df

st.set_page_config(page_title="Dashboard NeuroGuía v08", page_icon="📊", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #faf8ff 0%, #f7f6fb 100%); }
.dash-hero {
  background: linear-gradient(135deg, rgba(124,58,237,.12), rgba(59,130,246,.08));
  border: 1px solid rgba(124,58,237,.12);
  border-radius: 22px;
  padding: 1.1rem 1.3rem;
  margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

init_db()

st.markdown("""
<div class="dash-hero">
  <h1 style="margin:.1rem 0 0 0;">Dashboard de NeuroGuía v08</h1>
  <p style="margin:.35rem 0 0 0;color:#6b7280;">
    Panel premium para tesis: uso, memoria segura, base curada, mejora periódica y monitoreo de sesiones.
  </p>
</div>
""", unsafe_allow_html=True)

def safe_count(query: str) -> int:
    df = read_sql_df(query)
    return int(df.iloc[0]["total"]) if not df.empty else 0

usuarios_total = safe_count("SELECT COUNT(*) AS total FROM usuarios")
sesiones_total = safe_count("SELECT COUNT(*) AS total FROM sesiones")
interacciones_total = safe_count("SELECT COUNT(*) AS total FROM interacciones")
memorias_total = safe_count("SELECT COUNT(*) AS total FROM memoria_usuario")
documentos_total = safe_count("SELECT COUNT(*) AS total FROM base_conocimiento")
mejora_total = safe_count("SELECT COUNT(*) AS total FROM analitica_mejora")

m1, m2, m3 = st.columns(3)
m1.metric("Usuarios", usuarios_total)
m2.metric("Sesiones", sesiones_total)
m3.metric("Interacciones", interacciones_total)
m4, m5, m6 = st.columns(3)
m4.metric("Memorias activas", memorias_total)
m5.metric("Documentos curados", documentos_total)
m6.metric("Registros de mejora", mejora_total)

roles = read_sql_df("SELECT COALESCE(rol_usuario, 'sin dato') AS categoria, COUNT(*) AS total FROM usuarios GROUP BY COALESCE(rol_usuario, 'sin dato') ORDER BY total DESC")
emociones = read_sql_df("SELECT COALESCE(emocion_detectada, 'sin dato') AS categoria, COUNT(*) AS total FROM interacciones GROUP BY COALESCE(emocion_detectada, 'sin dato') ORDER BY total DESC")
crisis = read_sql_df("SELECT COALESCE(crisis_tipo, 'sin dato') AS categoria, COUNT(*) AS total FROM interacciones GROUP BY COALESCE(crisis_tipo, 'sin dato') ORDER BY total DESC")
intenciones = read_sql_df("SELECT COALESCE(intencion_detectada, 'sin dato') AS categoria, COUNT(*) AS total FROM analitica_mejora GROUP BY COALESCE(intencion_detectada, 'sin dato') ORDER BY total DESC")
temas = read_sql_df("SELECT COALESCE(tema_detectado, 'sin dato') AS categoria, COUNT(*) AS total FROM analitica_mejora GROUP BY COALESCE(tema_detectado, 'sin dato') ORDER BY total DESC")
cierres = read_sql_df("SELECT COALESCE(percepcion_cambio, 'sin dato') AS categoria, COUNT(*) AS total FROM sesiones GROUP BY COALESCE(percepcion_cambio, 'sin dato') ORDER BY total DESC")
utilidad = read_sql_df("SELECT COALESCE(utilidad_percibida, 'sin dato') AS categoria, COUNT(*) AS total FROM sesiones GROUP BY COALESCE(utilidad_percibida, 'sin dato') ORDER BY total DESC")
memoria_cat = read_sql_df("SELECT COALESCE(categoria, 'sin dato') AS categoria, COUNT(*) AS total FROM memoria_usuario GROUP BY COALESCE(categoria, 'sin dato') ORDER BY total DESC")
historial = read_sql_df("SELECT created_at, rol_usuario, emocion_detectada, crisis_tipo, intent_detectado, topic_detectado, filtro_clinico FROM interacciones ORDER BY created_at DESC LIMIT 50")

def chart_block(df: pd.DataFrame, title: str):
    st.subheader(title)
    if df.empty:
        st.caption("Todavía no hay datos.")
        return
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("categoria:N", sort="-y", title=""),
            y=alt.Y("total:Q", title="Total"),
            tooltip=["categoria", "total"]
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    chart_block(roles, "Distribución por rol")
with c2:
    chart_block(emociones, "Emociones detectadas")

c3, c4 = st.columns(2)
with c3:
    chart_block(crisis, "Tipos de crisis")
with c4:
    chart_block(intenciones, "Intenciones detectadas")

c5, c6 = st.columns(2)
with c5:
    chart_block(temas, "Temas detectados")
with c6:
    chart_block(memoria_cat, "Memoria por categoría")

c7, c8 = st.columns(2)
with c7:
    chart_block(cierres, "Percepción de cambio")
with c8:
    chart_block(utilidad, "Utilidad percibida")

st.subheader("Tablas resumen")
t1, t2 = st.columns(2)
with t1:
    st.dataframe(temas.rename(columns={"categoria":"tema"}), use_container_width=True, hide_index=True)
with t2:
    st.dataframe(emociones.rename(columns={"categoria":"emocion"}), use_container_width=True, hide_index=True)

st.subheader("Historial reciente")
if not historial.empty:
    st.dataframe(historial, use_container_width=True, hide_index=True)
else:
    st.caption("Todavía no hay interacciones guardadas.")
