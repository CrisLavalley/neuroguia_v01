# NeuroGuía v12 — arquitectura para tesis

## Qué contiene
- Orquestador propio (`src/orchestrator.py`)
- Cliente (`src/llm_client.py`)
- Constructor de prompts controlados (`src/prompt_builder.py`)
- Interfaz Streamlit simplificada (`app.py`)
- Dashboard básico (`app_dashboard.py`)
- Fallback determinista
- Límite no clínico y detección de riesgo

## Cómo funciona
1. NeuroGuía detecta perfil, tema, intención, riesgo y suficiencia contextual.
2. Recupera memoria y conocimiento curado.
3. Construye un prompt controlado.
4. Usa un fallback local.
5. Guarda trazabilidad e interacción.

## Streamlit secrets sugeridos
```toml
SHOW_DEBUG_PANEL = false
# DATABASE_URL = ""
```

## Nota importante para tesis
El aporte no es diseñar NeuroGuía como sistema especializado,
con memoria, conocimiento curado, seguridad no clínica, personalización y trazabilidad.

## Interfaz UI v13
Esta versión añade una interfaz más visual y profesional, inspirada en dashboards conversacionales:
- encabezado tipo producto real
- métricas superiores
- chat central amplio
- panel derecho con estado actual
- barra lateral limpia
