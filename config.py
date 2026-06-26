"""
config.py — Configuración global de Aria 1.0 (motor Gemini 2.5 Flash).

Toda variable ajustable del sistema vive aquí. Filosofía de esta versión:
RAPIDEZ + CALIDAD. Captura en RAM (cero disco), historial mínimo, respuestas
cortas y deterministas del modelo.

La API key NO se incrusta aquí: se lee de la variable de entorno GEMINI_API_KEY,
que se carga desde el archivo .env (ignorado por git). Si no está definida, el
sistema falla al arrancar con un mensaje claro.
"""

import os

# ─── Identidad ────────────────────────────────────────────────────────────────
AGENT_NAME    = "Aria"
AGENT_VERSION = "1.0.0"

# Directorio base del proyecto (este archivo).
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _cargar_dotenv(ruta: str) -> None:
    """
    Cargador .env mínimo (sin dependencias): vuelca las claves KEY=VALUE del
    archivo en os.environ SIN pisar variables ya definidas en el entorno real
    (estas tienen prioridad). Ignora comentarios (#) y líneas vacías. Si el
    archivo no existe, no hace nada.
    """
    if not os.path.isfile(ruta):
        return
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea or linea.startswith("#") or "=" not in linea:
                    continue
                clave, _, valor = linea.partition("=")
                valor = valor.strip().strip('"').strip("'")
                os.environ.setdefault(clave.strip(), valor)
    except OSError:
        pass


# Carga el .env antes de leer cualquier variable de entorno.
_cargar_dotenv(os.path.join(_BASE_DIR, ".env"))

# ─── Gemini (Google AI Studio) ────────────────────────────────────────────────
# Modelo multimodal nativo: procesa imagen + texto en un solo flujo.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    raise RuntimeError("Configura GEMINI_API_KEY en el archivo .env")

GEMINI_MODEL   = os.getenv("ARIA_MODEL", "gemini-2.5-flash")
GEMINI_BASE    = "https://generativelanguage.googleapis.com/v1beta"
# URL final del endpoint generateContent (se construye con el modelo activo).
GEMINI_URL     = f"{GEMINI_BASE}/models/{GEMINI_MODEL}:generateContent"

# ─── Generación del modelo ────────────────────────────────────────────────────
# Temperatura 0 → acción reproducible y de baja latencia. Respuestas cortas: el
# formato PENSAMIENTO/ACCION/FIN cabe de sobra en pocos tokens.
GEN_TEMPERATURE   = 0.0
GEN_TOP_P         = 0.9
GEN_MAX_TOKENS    = 256            # tope de tokens VISIBLES de respuesta por turno
GEN_STOP          = ["FIN"]        # corta la generación en el terminador del formato

# ─── Presupuesto de razonamiento (mapeado a la FSM) ───────────────────────────
# Gemini 2.5 Flash es un modelo "thinking". Para máxima velocidad lo desactivamos
# (budget 0) en estado WORKING; en THINKING le damos margen para problemas duros.
# En OVERLOADED (PC saturado) se fuerza a 0 para reducir la profundidad de cálculo.
THINK_BUDGET_RAPIDO   = 0          # WORKING / acción directa → sin pensamiento extra
THINK_BUDGET_PROFUNDO = 2048       # THINKING → razonamiento para problemas complejos
# Cuando hay budget de pensamiento, la respuesta visible necesita margen aparte.
THINK_RESPUESTA_EXTRA = 256

# ─── Imagen / captura (todo en RAM, sin disco) ────────────────────────────────
# Con Gemini en la nube ya no hay límite de VRAM local: subimos la resolución para
# que las coordenadas de clic sean más precisas, sin pasarnos (tokens/latencia).
IMG_MAX_SIZE   = 1280      # lado más largo de la captura enviada al modelo (px)
JPEG_QUALITY   = 75        # calidad JPEG al recodificar la captura
MONITOR_INDEX  = 1         # 1 = monitor principal (mss); 0 = todos combinados

# ─── Timeouts de red (segundos) ───────────────────────────────────────────────
TIMEOUT_CONNECT = 10.0
TIMEOUT_READ    = 60.0
TIMEOUT_WRITE   = 20.0
TIMEOUT_POOL    = 65.0

# ─── Bucle de tarea autónoma ──────────────────────────────────────────────────
MAX_PASOS_TAREA   = 12     # tope de ciclos visión→acción por comando (anti-bucle)
MAX_HISTORIAL     = 8       # turnos de historial conservados (historial mínimo)
DELAY_ESTABILIDAD = 1.0    # s: espera tras una acción para que Windows repinte

# ─── Telemetría / estado OVERLOADED ───────────────────────────────────────────
# Si CPU o RAM superan estos umbrales, la FSM pasa a OVERLOADED y reduce la
# profundidad de razonamiento (budget 0 + pausa defensiva).
CPU_OVERLOAD_PCT  = 92.0
RAM_OVERLOAD_PCT  = 92.0
TEMP_OVERLOAD_C   = 90.0   # °C (si el sensor está disponible; None si no)
PAUSA_OVERLOADED  = 5.0    # s de espera defensiva antes de reintentar

# ─── Parada limpia / persistencia de estado ───────────────────────────────────
STATE_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "aria_state.json")

# ─── Avatar VTuber (opcional, no crítico) ─────────────────────────────────────
VTUBE_ENABLED          = True       # si False, ni se intenta conectar
VTUBE_WS_URL           = "ws://localhost:8001"
VTUBE_PLUGIN_NAME      = "Aria 1.0 Agent"
VTUBE_PLUGIN_DEV       = "Aria"
VTUBE_TOKEN_FILE       = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "vtube_token.txt")
VTUBE_TIMEOUT_CONECTAR = 3
VTUBE_TIMEOUT_RECIBIR  = 5
VTUBE_TIMEOUT_APROBAR  = 30

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("ARIA_LOG_LEVEL", "INFO")
