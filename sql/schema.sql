CREATE TABLE IF NOT EXISTS usuarios (
    id_usuario TEXT PRIMARY KEY,
    email_usuario TEXT,
    nombre_mostrado TEXT,
    rol_usuario TEXT,
    estado_referencia TEXT,
    red_apoyo TEXT,
    consentimiento_menor BOOLEAN,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS sesiones (
    id_sesion TEXT PRIMARY KEY,
    id_usuario TEXT,
    email_usuario TEXT,
    rol_usuario TEXT,
    estado_referencia TEXT,
    opened_at TEXT,
    closed_at TEXT,
    percepcion_cambio TEXT,
    utilidad_percibida TEXT,
    status TEXT
);

CREATE TABLE IF NOT EXISTS interacciones (
    id_interaccion TEXT PRIMARY KEY,
    session_id TEXT,
    id_usuario TEXT,
    email_usuario TEXT,
    rol_usuario TEXT,
    estado_referencia TEXT,
    turno INTEGER,
    mensaje_usuario TEXT,
    emocion_detectada TEXT,
    probabilidad REAL,
    intensidad REAL,
    crisis_tipo TEXT,
    filtro_clinico TEXT,
    protocolo TEXT,
    respuesta_ia TEXT,
    fuente_respuesta TEXT,
    intent_detectado TEXT,
    topic_detectado TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS memoria_usuario (
    id_memoria TEXT PRIMARY KEY,
    id_usuario TEXT,
    categoria TEXT,
    clave TEXT,
    valor TEXT,
    nivel_confianza REAL,
    fuente TEXT,
    ultima_actualizacion TEXT,
    activo BOOLEAN
);
