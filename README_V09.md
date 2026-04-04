# NeuroGuía v09 — versión final funcional

## Qué incluye
- Interfaz simplificada y más cálida
- Uso real del nombre del usuario en saludo y acompañamiento
- Respuestas menos rígidas y menos repetitivas
- Adaptación del tono según perfil (madre, padre, abuelo(a), cuidador(a), etc.)
- Consulta opcional de información general en internet mediante Wikipedia
- Límite no clínico reforzado
- Dashboard con gráficas
- Debug oculto para usuarios normales

## Estructura
- `app.py`
- `app_dashboard.py`
- `src/auth_helpers.py`
- `src/database.py`
- `src/neuroguia_engine.py`
- `src/web_helper.py`
- `sql/schema.sql`
- `knowledge_base/rag_documents.json`
- `knowledge_base/protocolos_intervencion.json`

## Secrets opcionales
```toml
SHOW_DEBUG_PANEL = false
# DATABASE_URL = "postgresql+psycopg://usuario:password@host:puerto/db"
```

## Recomendación de despliegue
Si solo se busca estabilidad para tesis y demo:
- NO definir `DATABASE_URL`
- dejar que la app use SQLite local automáticamente

## Nota ética
NeuroGuía acompaña, orienta y ayuda a ordenar lo observado.
No sustituye atención clínica ni evaluación profesional.
