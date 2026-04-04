
CREATE TABLE IF NOT EXISTS usuarios (
    id_usuario TEXT PRIMARY KEY,
    email_usuario TEXT,
    nombre_mostrado TEXT,
    rol_usuario TEXT,
    estado_referencia TEXT,
    red_apoyo TEXT,
    consentimiento_menor BOOLEAN DEFAULT FALSE,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS sesiones (
    id_sesion TEXT PRIMARY KEY,
    id_usuario TEXT NOT NULL,
    email_usuario TEXT,
    rol_usuario TEXT,
    estado_referencia TEXT,
    opened_at TEXT,
    closed_at TEXT,
    status TEXT,
    percepcion_cambio TEXT,
    utilidad_percibida TEXT
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
    probabilidad DOUBLE PRECISION,
    intensidad DOUBLE PRECISION,
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
    id_usuario TEXT NOT NULL,
    categoria TEXT NOT NULL,
    clave TEXT NOT NULL,
    valor TEXT,
    nivel_confianza DOUBLE PRECISION,
    fuente TEXT,
    ultima_actualizacion TEXT,
    activo BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS base_conocimiento (
    id_documento TEXT PRIMARY KEY,
    titulo TEXT NOT NULL,
    tema TEXT,
    subtema TEXT,
    contenido TEXT,
    fuente TEXT,
    tipo_fuente TEXT,
    fecha_ingreso TEXT,
    nivel_confianza DOUBLE PRECISION,
    validado BOOLEAN DEFAULT TRUE,
    palabras_clave TEXT,
    visible_para TEXT
);

CREATE TABLE IF NOT EXISTS analitica_mejora (
    id_registro TEXT PRIMARY KEY,
    id_interaccion TEXT,
    tema_detectado TEXT,
    intencion_detectada TEXT,
    respuesta_util TEXT,
    tipo_fallo TEXT,
    hubo_seguimiento BOOLEAN,
    hubo_cierre BOOLEAN,
    percepcion_cambio TEXT,
    utilidad_percibida TEXT,
    requiere_revision BOOLEAN DEFAULT FALSE,
    observaciones_revision TEXT,
    version_motor TEXT
);

CREATE TABLE IF NOT EXISTS incidentes_respuesta (
    id_incidente TEXT PRIMARY KEY,
    id_interaccion TEXT,
    categoria_incidente TEXT,
    descripcion TEXT,
    severidad TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS versiones_motor (
    id_version TEXT PRIMARY KEY,
    nombre_version TEXT,
    descripcion TEXT,
    fecha_lanzamiento TEXT
);
