# Arquitectura híbrida de NeuroGuía para tesis

## Definición
NeuroGuía se concibe como un sistema conversacional híbrido de apoyo no clínico. Su valor no reside en el modelo de lenguaje en sí, sino en la arquitectura propia que orquesta seguridad, memoria contextual, recuperación de conocimiento validado, personalización por perfil y trazabilidad analítica.

## Capas
### Capa 1. Interfaz
Permite la interacción conversacional, el registro del perfil y la captura de utilidad percibida.

### Capa 2. Orquestador
Detecta perfil, intención, tema, nivel de riesgo y suficiencia contextual. Decide si la consulta puede responderse, si requiere límite no clínico o si debe redirigirse.

### Capa 3. Memoria y base curada
Recupera temas frecuentes, detonantes y documentos validados para enriquecer la respuesta.

### Capa 4. Generación lingüística
Usa un LLM únicamente como motor de redacción controlada, bajo un prompt restringido por reglas de seguridad y tono.

### Capa 5. Analítica
Registra interacción, tema, intención, utilidad y memoria para evaluación y mejora.

## Justificación metodológica
Este enfoque evita depender de respuestas rígidas totalmente programadas y, al mismo tiempo, evita delegar el control total al modelo de lenguaje. Así, la inteligencia generativa queda subordinada a la lógica de NeuroGuía, que impone límites no clínicos, contextualización y personalización.

## Aporte de tesis
El aporte está en el diseño del sistema híbrido, no en la creación de un modelo base. Se propone una arquitectura especializada y segura para acompañamiento no clínico en contextos de neurodivergencia.
