# NeuroGuía v10 — versión optimizada

## Ajustes principales
- corrección del error de sintaxis del motor conversacional
- interfaz principal más limpia y menos saturada
- uso real del nombre del usuario
- respuestas más humanas y menos repetitivas
- adaptación del tono según perfil
- consulta opcional de información general en internet mediante Wikipedia
- límite no clínico reforzado
- dashboard con gráficas
- debug oculto para usuarios normales

## Secrets opcionales
```toml
SHOW_DEBUG_PANEL = false
# DATABASE_URL = "postgresql+psycopg://usuario:password@host:puerto/db"
```

## Recomendación de despliegue
Si solo buscas estabilidad para tesis y demo:
- no definir `DATABASE_URL`
- dejar que la app use SQLite local automáticamente
