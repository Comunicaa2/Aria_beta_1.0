"""
utils/image.py — Captura de pantalla en RAM para Aria 1.0.

REGLA DE ORO: CERO DISCO. Todo el pipeline vive en memoria:
    mss.grab() → bytes BGRA → PIL RGB → resize ≤ IMG_MAX_SIZE → JPEG → Base64.
Esto evita cuellos de Disk I/O y los PermissionError típicos de Windows al
reescribir la misma imagen en bucle.

La `Captura` lleva metadatos de ESCALA para mapear las coordenadas del espacio
IMAGEN (lo que ve Gemini) al espacio REAL de la pantalla (donde se hace el clic).

RESILIENCIA: si mss o Pillow no están, `disponible()` devuelve False y
`capturar()` devuelve None sin lanzar; el orquestador degrada a sin-visión.
"""

import base64
import io
import logging
import time
from dataclasses import dataclass
from typing import Optional

from config import IMG_MAX_SIZE, JPEG_QUALITY, MONITOR_INDEX

logger = logging.getLogger("aria.image")

# ─── Importación resiliente de dependencias ──────────────────────────────────
try:
    import mss as _mss
    _MSS_OK = True
except Exception as _exc:                       # noqa: BLE001
    _mss = None                                  # type: ignore[assignment]
    _MSS_OK = False
    logger.warning("mss no disponible (%s) — la visión quedará inactiva.", _exc)

try:
    from PIL import Image
    _PIL_OK = True
except Exception as _exc:                       # noqa: BLE001
    Image = None                                 # type: ignore[assignment]
    _PIL_OK = False
    logger.warning("Pillow no disponible (%s) — la visión quedará inactiva.", _exc)


@dataclass
class Captura:
    """
    Captura de pantalla + metadatos de escala IMAGEN→PANTALLA.

    Como el downscale conserva la proporción, escala_x ≈ escala_y, pero se
    exponen ambos por robustez. `real()` mapea un punto del espacio imagen al
    espacio real de la pantalla (donde pyautogui hace clic).
    """
    b64: str
    ancho_real: int
    alto_real: int
    ancho_img: int
    alto_img: int

    @property
    def escala_x(self) -> float:
        return self.ancho_real / self.ancho_img if self.ancho_img else 1.0

    @property
    def escala_y(self) -> float:
        return self.alto_real / self.alto_img if self.alto_img else 1.0

    def real(self, x_img: int, y_img: int) -> tuple[int, int]:
        """Convierte un punto del espacio IMAGEN al espacio REAL de la pantalla."""
        return int(round(x_img * self.escala_x)), int(round(y_img * self.escala_y))


def disponible() -> bool:
    """True si el stack de visión (mss + Pillow) está operativo."""
    return _MSS_OK and _PIL_OK


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLA DE ARIA — apartarla de la captura (Windows, ctypes puro)
# ══════════════════════════════════════════════════════════════════════════════
# La consola donde corre Aria TAPA el escritorio en la captura: el modelo se vería
# a sí mismo en vez de la app objetivo. Antes de percibir la minimizamos; al
# terminar la tarea se restaura. Resiliente: si no es Windows / falla → no-op.

_SW_MINIMIZE = 6
_SW_RESTORE  = 9
_SW_SHOWNOACTIVATE = 4   # restaura SIN activar/traer-al-frente (no tapa la app)


# HWND cacheado de la ventana de terminal ancestro (Windows Terminal / ConPTY).
_HWND_TERMINAL_CACHE: list = []          # [] = sin buscar; [hwnd|None] = resuelto


def _hwnd_terminal_ancestro():
    """Bajo Windows Terminal, GetConsoleWindow devuelve la ventana OCULTA del host
    ConPTY: minimizarla no hace nada y la consola sale en las capturas. Se busca la
    ventana visible del proceso WindowsTerminal.exe ancestro (vía psutil). Cachea el
    resultado (la terminal no cambia durante la sesión). None si no aplica."""
    if _HWND_TERMINAL_CACHE:
        return _HWND_TERMINAL_CACHE[0]
    hwnd_term = None
    try:
        import ctypes
        from ctypes import wintypes
        import psutil
        pids = {p.pid for p in psutil.Process().parents()
                if p.name().lower() == "windowsterminal.exe"}
        if pids:
            u32 = ctypes.windll.user32
            encontrado = []

            @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
            def _cb(hwnd, _lp):
                pid = wintypes.DWORD()
                u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if pid.value in pids and u32.IsWindowVisible(hwnd):
                    encontrado.append(hwnd)
                    return False                 # detener enumeración
                return True

            u32.EnumWindows(_cb, 0)
            hwnd_term = encontrado[0] if encontrado else None
    except Exception:                            # noqa: BLE001
        hwnd_term = None
    _HWND_TERMINAL_CACHE.append(hwnd_term)
    return hwnd_term


def hwnd_consola():
    """HWND de la ventana VISIBLE que aloja la consola de Aria: la propia consola
    clásica (conhost) o, bajo Windows Terminal, la ventana del terminal ancestro.
    None si no se puede determinar. Lo usan minimizar/restaurar y el controller."""
    try:
        import ctypes
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd and ctypes.windll.user32.IsWindowVisible(hwnd):
            return hwnd
        return _hwnd_terminal_ancestro() or (hwnd or None)
    except Exception:                            # noqa: BLE001
        return None


def minimizar_consola(settle: float = 0.18) -> bool:
    """Minimiza la consola de Aria para que NO salga en la captura. Idempotente."""
    hwnd = hwnd_consola()
    if not hwnd:
        return False
    try:
        import ctypes
        u32 = ctypes.windll.user32
        if u32.IsIconic(hwnd):                   # ya minimizada → nada que hacer
            return True
        u32.ShowWindow(hwnd, _SW_MINIMIZE)
        if settle > 0:
            time.sleep(settle)                   # deja repintar el escritorio
        return True
    except Exception as exc:                      # noqa: BLE001
        logger.debug("minimizar_consola: fallo (%s) → no-op.", exc)
        return False


def restaurar_consola() -> bool:
    """Restaura la consola de Aria al terminar la tarea, SIN traerla al frente
    (SW_SHOWNOACTIVATE): así no tapa la app recién abierta cuando el verificador
    captura la pantalla. Resiliente."""
    hwnd = hwnd_consola()
    if not hwnd:
        return False
    try:
        import ctypes
        ctypes.windll.user32.ShowWindow(hwnd, _SW_SHOWNOACTIVATE)
        return True
    except Exception as exc:                      # noqa: BLE001
        logger.debug("restaurar_consola: fallo (%s) → no-op.", exc)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# CAPTURA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def capturar(
    max_size: int = IMG_MAX_SIZE,
    calidad: int = JPEG_QUALITY,
    monitor_index: int = MONITOR_INDEX,
) -> Optional[Captura]:
    """
    Captura el monitor principal y devuelve una `Captura` (Base64 + escala).
    Todo en RAM. Devuelve None si la visión no está disponible o falla.
    """
    if not disponible():
        logger.debug("capturar: stack de visión inactivo → None.")
        return None

    try:
        with _mss.mss() as sct:
            monitores = sct.monitors
            idx = monitor_index if 0 <= monitor_index < len(monitores) else (
                1 if len(monitores) > 1 else 0
            )
            shot = sct.grab(monitores[idx])
            # BGRA nativo → PIL RGB (conversión documentada mss → Pillow).
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

        ancho_real, alto_real = img.size
        ancho_img, alto_img = ancho_real, alto_real
        lado_mayor = max(ancho_real, alto_real)
        if lado_mayor > max_size:
            factor = max_size / float(lado_mayor)
            ancho_img = max(1, int(ancho_real * factor))
            alto_img = max(1, int(alto_real * factor))
            img = img.resize((ancho_img, alto_img), Image.LANCZOS, reducing_gap=2.0)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=calidad, optimize=True)
        datos = buf.getvalue()
        b64 = base64.b64encode(datos).decode("ascii")

        logger.info(
            "Captura %dx%d → %dx%d px, JPEG q=%d, %.0f KB.",
            ancho_real, alto_real, ancho_img, alto_img, calidad, len(datos) / 1024.0,
        )
        return Captura(
            b64=b64,
            ancho_real=ancho_real, alto_real=alto_real,
            ancho_img=ancho_img, alto_img=alto_img,
        )
    except Exception as exc:                      # noqa: BLE001
        logger.warning("Fallo al capturar la pantalla: %s — degradando a None.", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# ESTABILIDAD DE PANTALLA — espera adaptativa por diff de miniaturas
# ══════════════════════════════════════════════════════════════════════════════
# En vez de dormir un tiempo fijo tras cada acción, capturamos miniaturas grises
# diminutas (rápidas, en RAM) y comparamos consecutivas; cuando la UI se queda
# quieta (o vence el tope), seguimos. Más rápido y robusto que un sleep fijo.

_MINI_LADO = 64


def _mini(monitor_index: int = MONITOR_INDEX) -> Optional[bytes]:
    if not disponible():
        return None
    try:
        with _mss.mss() as sct:
            monitores = sct.monitors
            idx = monitor_index if 0 <= monitor_index < len(monitores) else (
                1 if len(monitores) > 1 else 0
            )
            shot = sct.grab(monitores[idx])
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        return img.convert("L").resize((_MINI_LADO, _MINI_LADO), Image.BILINEAR).tobytes()
    except Exception as exc:                      # noqa: BLE001
        logger.debug("_mini: fallo (%s) → None.", exc)
        return None


def _diff(a: Optional[bytes], b: Optional[bytes]) -> float:
    """Diferencia media normalizada [0..1] entre dos miniaturas (1.0 si faltan).
    Puro Python: a 64x64 px (4 KB) el bucle cuesta ~1 ms, no amerita numpy."""
    if not a or not b or len(a) != len(b):
        return 1.0
    total = sum(abs(x - y) for x, y in zip(a, b))
    return (total / len(a)) / 255.0


def esperar_estabilidad(
    max_seg: float = 2.5,
    intervalo: float = 0.18,
    umbral: float = 0.02,
    estables: int = 2,
    min_espera: float = 0.5,
) -> bool:
    """
    Espera adaptativa. PRIMERO duerme `min_espera` s (piso real para que la acción
    surta efecto antes de empezar a comparar — evita capturar el frame muerto previo
    al cambio), LUEGO captura miniaturas y vuelve cuando hay `estables` lecturas
    seguidas con diff < `umbral` (UI quieta) o se alcanza `max_seg`. Devuelve True
    si la pantalla se estabilizó. Sin visión → sleep corto y False. El piso es
    ADITIVO: `max_seg` es la ventana de muestreo DESPUÉS del piso.
    """
    time.sleep(max(0.0, min_espera))             # FIX #1: piso antes de comparar
    inicio = time.monotonic()
    prev = _mini()
    if prev is None:
        time.sleep(min(0.4, max_seg))
        return False

    seguidas = 0
    while (time.monotonic() - inicio) < max_seg:
        time.sleep(intervalo)
        actual = _mini()
        if actual is None:
            continue
        if _diff(prev, actual) < umbral:
            seguidas += 1
            if seguidas >= estables:
                return True
        else:
            seguidas = 0
        prev = actual
    return False
