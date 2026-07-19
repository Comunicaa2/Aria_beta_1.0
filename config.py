"""
config.py — Configuración global de Aria 1.0 (motor Gemini 3.5 Flash).

Toda variable ajustable del sistema vive aquí. Filosofía de esta versión:
RAPIDEZ + CALIDAD. Captura en RAM (cero disco), historial mínimo, respuestas
cortas y deterministas del modelo.

La API key NO se incrusta aquí: se lee de la variable de entorno GEMINI_API_KEY,
que se carga desde el archivo .env (ignorado por git). Si no está definida, el
sistema falla al arrancar con un mensaje claro.
"""

import os

from compartido import cargar_dotenv

# ─── Identidad ────────────────────────────────────────────────────────────────
AGENT_NAME    = "Aria"
AGENT_VERSION = "1.0.0"

# Directorio base del proyecto (este archivo).
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Carga el .env antes de leer cualquier variable de entorno.
cargar_dotenv(os.path.join(_BASE_DIR, ".env"))

# ─── Gemini (Google AI Studio) ────────────────────────────────────────────────
# Modelo multimodal nativo: procesa imagen + texto en un solo flujo.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    raise RuntimeError("Configura GEMINI_API_KEY en el archivo .env")

GEMINI_MODEL   = os.getenv("ARIA_MODEL", "gemini-3.5-flash")
GEMINI_BASE    = "https://generativelanguage.googleapis.com/v1beta"
# URL final del endpoint generateContent (se construye con el modelo activo).
GEMINI_URL     = f"{GEMINI_BASE}/models/{GEMINI_MODEL}:generateContent"

# ─── Modo entrenamiento + fallback NVIDIA NIM (TEMPORAL) ──────────────────────
# Interruptor único del fallback. En producción debe quedar en false: ante un 429
# Aria guarda estado y se detiene (sin fallback). El fallback a NIM SOLO se activa
# si TRAINING_MODE = true Y hay NVIDIA_API_KEY. Para retirar el fallback más tarde:
# pon ARIA_TRAINING_MODE=false (o borra el bloque de fallback en core/brain.py).
TRAINING_MODE  = os.getenv("ARIA_TRAINING_MODE", "false").lower() == "true"
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Cadena de fallback NIM, probada EN ORDEN tras un 429 de Gemini. Aria es un
# agente BASADO EN VISIÓN: el fallback primario debe poder ver la captura, o
# razona a ciegas (causa confirmada de tareas atascadas/expiradas en training).
#   1º meta/llama-3.2-90b-vision-instruct (CON visión, probado 4/4 consistente).
#   2º minimaxai/minimax-m3 (sin visión) como último recurso.
NVIDIA_MODEL_PRIMARIO   = os.getenv("ARIA_NVIDIA_MODEL",
                                    "meta/llama-3.2-90b-vision-instruct")
NVIDIA_MODEL_SECUNDARIO = os.getenv("ARIA_NVIDIA_MODEL_2",
                                    "minimaxai/minimax-m3")
NVIDIA_FALLBACK_MODELS  = [NVIDIA_MODEL_PRIMARIO, NVIDIA_MODEL_SECUNDARIO]

# ─── Generación del modelo ────────────────────────────────────────────────────
# Temperatura 0 → acción reproducible y de baja latencia. Respuestas cortas: el
# formato PENSAMIENTO/ACCION/FIN cabe de sobra en pocos tokens.
# Temperatura > 0 para romper bucles deterministas (FIX #11): ante una captura
# casi idéntica, una temperatura 0 repetía la misma acción inútil. 0.2 introduce
# variación suficiente sin perder reproducibilidad para clics/formato.
GEN_TEMPERATURE   = 0.2
GEN_TOP_P         = 0.9
GEN_MAX_TOKENS    = 2048           # tope de tokens VISIBLES: margen para el bloque
                                   # CONTENIDO de 'guardar' (archivos/scripts); las
                                   # respuestas normales siguen siendo cortas.
GEN_STOP          = ["FIN"]        # corta la generación en el terminador del formato

# ─── Presupuesto de razonamiento (mapeado a la FSM) ───────────────────────────
# Gemini 3.5 Flash es un modelo "thinking". Para máxima velocidad lo desactivamos
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
JPEG_QUALITY   = 85        # FIX #10: 75→85, menos artefactos para leer texto/iconos
MONITOR_INDEX  = 1         # 1 = monitor principal (mss); 0 = todos combinados

# ─── Timeouts de red (segundos) ───────────────────────────────────────────────
TIMEOUT_CONNECT = 10.0
TIMEOUT_READ    = 60.0
TIMEOUT_WRITE   = 20.0
TIMEOUT_POOL    = 65.0

# ─── Carpeta de trabajo (acciones guardar / ejecutar_python) ──────────────────
# Aquí escribe Aria sus archivos (informes, scripts de análisis) y desde aquí se
# ejecutan. Solo nombres de archivo simples: el controller rechaza rutas.
WORKSPACE_DIR  = os.path.join(_BASE_DIR, "workspace")
TIMEOUT_SCRIPT = 60        # s máximos de un ejecutar_python antes de cancelarlo

# ─── Bucle de tarea autónoma ──────────────────────────────────────────────────
MAX_PASOS_TAREA   = 12     # tope de ciclos visión→acción por comando (anti-bucle)
MAX_HISTORIAL     = 12      # FIX #2: 8→12 (~4 ciclos). Las imágenes viejas ya se
                           # podan en _gc_imagenes, así que el coste en tokens es bajo.
DELAY_ESTABILIDAD = 1.0    # s: espera tras una acción para que Windows repinte

# ─── Telemetría / estado OVERLOADED ───────────────────────────────────────────
# Si CPU o RAM superan estos umbrales, la FSM pasa a OVERLOADED y reduce la
# profundidad de razonamiento (budget 0 + pausa defensiva).
CPU_OVERLOAD_PCT  = 92.0
RAM_OVERLOAD_PCT  = 92.0
TEMP_OVERLOAD_C   = 90.0   # °C (si el sensor está disponible; None si no)
PAUSA_OVERLOADED  = 5.0    # s de espera defensiva antes de reintentar

# ─── Modo AHORRO DE ENERGÍA (comportamiento del estado OVERLOADED) ────────────
# Con el PC saturado, Aria reduce su PROPIO consumo en vez de solo esperar:
#   · captura más pequeña y más comprimida (menos CPU, red y tokens),
#   · razonamiento a budget 0 (ya mapeado arriba),
#   · pausa defensiva (PAUSA_OVERLOADED),
#   · avatar a ritmo lento (1 Hz en vez de 15 Hz).
AHORRO_IMG_SIZE = 960      # px lado mayor de la captura en ahorro (normal: IMG_MAX_SIZE)
AHORRO_JPEG_Q   = 70       # calidad JPEG en ahorro (normal: JPEG_QUALITY)

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
