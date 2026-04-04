# NeuroGuía v12 — arquitectura híbrida para tesis

## Qué contiene
- Orquestador propio (`src/orchestrator.py`)
- Cliente opcional para OpenAI API (`src/llm_client.py`)
- Constructor de prompts controlados (`src/prompt_builder.py`)
- Interfaz Streamlit simplificada (`app.py`)
- Dashboard básico (`app_dashboard.py`)
- Fallback determinista cuando no hay API key
- Límite no clínico y detección de riesgo

## Cómo funciona
1. NeuroGuía detecta perfil, tema, intención, riesgo y suficiencia contextual.
2. Recupera memoria y conocimiento curado.
3. Construye un prompt controlado.
4. Si existe `OPENAI_API_KEY` y `OPENAI_MODEL`, consulta al LLM.
5. Si no existe API, usa un fallback local.
6. Guarda trazabilidad e interacción.

## Variables de entorno para modo híbrido
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_TEMPERATURE` (opcional)
- `OPENAI_TIMEOUT` (opcional)

## Streamlit secrets sugeridos
```toml
SHOW_DEBUG_PANEL = false
# DATABASE_URL = ""
```

## Nota importante para tesis
El aporte no es “inventar un LLM”, sino diseñar NeuroGuía como sistema híbrido especializado,
con memoria, conocimiento curado, seguridad no clínica, personalización y trazabilidad.
