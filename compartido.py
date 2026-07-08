"""
compartido.py — Código común a Aria y al Entrenador (procesos separados).

El desacople REAL entre ambos procesos vive en los archivos de tasks/ (cola,
stats, marca de rate limit); compartir este módulo no acopla sus ciclos de
vida. Antes estas piezas estaban duplicadas a mano en core/brain.py, config.py
y trainer/gemini_cliente.py con comentarios "deben coincidir" — ahora hay una
sola fuente de verdad.

Contiene:
  · Identificadores de modelo (GEMINI / NIM-LLAMA / NIM-MINIMAX).
  · Cargador .env mínimo (sin dependencias).
  · Rate limiter CROSS-PROCESS de Gemini (archivo de marca + mutex O_EXCL).
"""

import json
import logging
import os
import time

logger = logging.getLogger("compartido")

_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─── Identificadores de modelo (logs de Aria, panel y stats del entrenador) ───
MODELO_GEMINI    = "GEMINI"
MODELO_NIM_LLAMA = "NIM-LLAMA"   # meta/llama-3.2-90b-vision-instruct
MODELO_NIM_OMNI  = "NIM-MINIMAX"  # minimaxai/minimax-m3


def cargar_dotenv(ruta: str) -> None:
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


# ─── Rate limiter CROSS-PROCESS de Gemini ─────────────────────────────────────
# Aria y el entrenador comparten la cuota de 15 RPM de la MISMA API key. Se
# coordinan por un archivo de marca (tasks/rate_gemini.json) protegido con un
# mutex de exclusión mutua (lock O_EXCL): antes de cada llamada a Gemini se
# espera a que hayan pasado ≥ GEMINI_MIN_INTERVALO s desde la última llamada de
# CUALQUIER proceso. Usa time.time() (reloj de pared, comparable entre
# procesos). Si el archivo no está disponible, cae a un piso en memoria (no
# rompe el comportamiento).
GEMINI_MIN_INTERVALO = 4.5   # 15 RPM = 1 req/4s; margen de seguridad sobre los 4s

_RATE_DIR  = os.path.join(_ROOT, "tasks")
_RATE_FILE = os.path.join(_RATE_DIR, "rate_gemini.json")
_RATE_LOCK = os.path.join(_RATE_DIR, "rate_gemini.lock")
_RATE_LOCK_TIMEOUT = 15.0    # s máx esperando el mutex antes de degradar a local
_RATE_LOCK_STALE   = 12.0    # s: un lock más viejo es huérfano (proceso caído) → robar
_rate_ultima_local = 0.0     # piso en memoria (monotonic) si el archivo falla


def _rate_lock_adquirir() -> bool:
    """Mutex inter-proceso por archivo (O_EXCL). True si lo adquirió."""
    inicio = time.monotonic()
    while True:
        try:
            fd = os.open(_RATE_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            try:                                   # robar lock huérfano (proceso caído)
                if time.time() - os.path.getmtime(_RATE_LOCK) > _RATE_LOCK_STALE:
                    os.remove(_RATE_LOCK)
                    continue
            except OSError:
                pass
            if time.monotonic() - inicio > _RATE_LOCK_TIMEOUT:
                return False
            time.sleep(0.05)
        except OSError:
            return False


def _rate_lock_liberar() -> None:
    try:
        os.remove(_RATE_LOCK)
    except OSError:
        pass


def rate_limit_compartido() -> None:
    """
    Garantiza ≥ GEMINI_MIN_INTERVALO s entre llamadas a Gemini de CUALQUIER
    proceso (Aria + entrenador), vía el archivo de marca bajo mutex. Atómico:
    la sección leer→esperar→escribir corre dentro del lock. Degrada a piso en
    memoria si el sistema de archivos no está disponible (sin romper nada).
    """
    global _rate_ultima_local
    try:
        os.makedirs(_RATE_DIR, exist_ok=True)
    except OSError:
        pass

    if not _rate_lock_adquirir():                  # no se pudo coordinar → piso local
        logger.error("Rate limit: NO se pudo adquirir el mutex en %.0fs — degradando a "
                     "piso LOCAL; los procesos podrían exceder %.0f RPM combinados.",
                     _RATE_LOCK_TIMEOUT, 60.0 / GEMINI_MIN_INTERVALO)
        espera = GEMINI_MIN_INTERVALO - (time.monotonic() - _rate_ultima_local)
        if espera > 0:
            logger.info("Rate limit (local): esperando %.1fs", espera)
            time.sleep(espera)
        _rate_ultima_local = time.monotonic()
        return

    try:
        ultima = 0.0
        try:
            with open(_RATE_FILE, "r", encoding="utf-8") as f:
                ultima = float(json.load(f).get("ultima_llamada", 0.0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            ultima = 0.0
        # Clamp ante saltos del reloj de pared (NTP): nunca esperar más del intervalo.
        espera = max(0.0, min(GEMINI_MIN_INTERVALO - (time.time() - ultima),
                              GEMINI_MIN_INTERVALO))
        if espera > 0:
            logger.info("Rate limit (compartido): esperando %.1fs", espera)
            time.sleep(espera)
        try:                                       # marca esta llamada (escritura atómica)
            tmp = _RATE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"ultima_llamada": time.time()}, f)
            os.replace(tmp, _RATE_FILE)
        except OSError:
            pass
        _rate_ultima_local = time.monotonic()
    finally:
        _rate_lock_liberar()
