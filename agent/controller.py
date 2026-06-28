"""
agent/controller.py — Capa de automatización (mouse + teclado) de Aria 1.0.

Recibe la cadena de ACCION cruda que decidió Gemini y la ejecuta en el SO.
Comandos válidos (uno por turno):

    click X Y           clic izquierdo en (X, Y) del espacio IMAGEN
    double_click X Y    doble clic en (X, Y) del espacio IMAGEN
    type TEXTO          escribe el texto exacto
    key TECLA           pulsa una tecla       (key enter / key esc / key tab)
    hotkey A+B[+C]      combinación           (hotkey ctrl+c / hotkey ctrl+s)
    launch_app NOMBRE   lanza una app por nombre (launch_app notepad / calc / msedge)
    wait N              espera N segundos
    done                señala tarea completada (no toca el SO)

Las coordenadas que da el modelo están en el espacio de la IMAGEN reducida; este
módulo las reescala al espacio REAL de la pantalla usando la `Captura`.

Envuelve pyautogui de forma resiliente: si no está disponible, entra en modo
SIMULACIÓN (registra la acción pero no la ejecuta) y nunca lanza hacia arriba.
"""

import logging
import os
import re
import time
from typing import Optional

from utils.image import Captura

logger = logging.getLogger("aria.controller")

try:
    import pyautogui
    pyautogui.FAILSAFE = True      # mover el ratón a una esquina aborta (seguridad)
    pyautogui.PAUSE = 0.03
    _OK = True
except Exception as _exc:                          # noqa: BLE001
    pyautogui = None                               # type: ignore[assignment]
    _OK = False
    logger.warning("Controller en SIMULACIÓN (pyautogui no disponible: %s).", _exc)

# Movimiento verificado del cursor (paridad con la v0.3): el cursor viaja visible
# y la acción solo se dispara si llegó de verdad al destino.
_DUR_MOV       = 0.22
_TOL_PX        = 3
_ESPERA_REINT  = 0.18
_WAIT_MAX      = 30.0

# Unidades de rueda por "clic" de scroll. En Windows un notch del ratón = 120
# (WHEEL_DELTA); pyautogui.scroll() recibe esas unidades crudas, así que cada N
# equivale a N notches reales de la rueda.
_SCROLL_PASO   = 120
_SCROLL_DEF    = 3          # N por defecto si no se especifica
_SCROLL_MAX    = 30         # tope defensivo de clics (el modelo a veces pide N enorme)

# Margen de seguridad anti-esquina. pyautogui.FAILSAFE aborta TODO el programa si
# el cursor llega a una esquina de la pantalla; el modelo a veces pide coords como
# (0, 720). Rechazamos cualquier destino dentro de este margen de las 4 esquinas
# ANTES de mover el ratón — sin desactivar FAILSAFE (la protección sigue intacta).
_MARGEN_ESQUINA = 60        # px a cada lado de cada esquina (zona prohibida)

# Combos que abren un diálogo del SO: tras enviarlos hay que esperar a que la
# ventana aparezca antes de escribir, o el siguiente 'type' cae al vacío.
_COMBOS_DIALOGO = {
    frozenset({"win", "r"}): 0.7,    # diálogo "Ejecutar"
    frozenset({"win"}):      0.4,    # menú Inicio
}

# ─── Lista negra de hotkeys (seguridad) ───────────────────────────────────────
# Combinaciones que cierran/secuestran el sistema o rompen la visión de Aria. Se
# bloquean ANTES de ejecutarse. Normalizadas: el orden de teclas y las mayúsculas
# NO importan. Para editar, añade/quita un frozenset aquí (único lugar).
_HOTKEYS_PROHIBIDAS = frozenset({
    frozenset({"alt", "f4"}),            # cierra la ventana activa
    frozenset({"win", "l"}),             # bloquea la sesión
    frozenset({"win", "d"}),             # muestra escritorio (minimiza todo)
    frozenset({"win", "r"}),             # Ejecutar (usa launch_app en su lugar)
    frozenset({"win", "alt", "r"}),      # grabadora de pantalla
    frozenset({"win", "g"}),             # Xbox Game Bar
    frozenset({"ctrl", "alt", "del"}),   # pantalla de seguridad
    frozenset({"win", "x"}),             # menú de usuario avanzado
    frozenset({"alt", "tab"}),           # cambiador de ventanas
    frozenset({"win", "tab"}),           # vista de tareas
    frozenset({"ctrl", "w"}),            # cierra pestaña/ventana (pierde trabajo)
    frozenset({"ctrl", "shift", "esc"}), # Administrador de tareas
    frozenset({"f11"}),                  # pantalla completa (rompe la visión)
})

# Alias → nombre canónico pyautogui, para que la lista negra sea robusta a variantes.
_ALIAS_TECLAS = {
    "control": "ctrl", "windows": "win", "super": "win", "cmd": "win",
    "escape": "esc", "delete": "del", "supr": "del",
}


def _es_hotkey_prohibida(norm: list[str]) -> bool:
    """True si la combinación (normalizada por alias) está en la lista negra."""
    combo = frozenset(_ALIAS_TECLAS.get(t, t) for t in norm)
    return combo in _HOTKEYS_PROHIBIDAS

# ─── Patrones de comandos ─────────────────────────────────────────────────────
_RE_CLICK        = re.compile(r"^click\s+(-?\d+)\s+(-?\d+)\s*$",        re.I)
_RE_DOUBLE_CLICK = re.compile(r"^double_click\s+(-?\d+)\s+(-?\d+)\s*$", re.I)
_RE_TYPE         = re.compile(r"^type\s+(.+)$",                          re.I | re.S)
_RE_KEY          = re.compile(r"^key\s+(\S+)\s*$",                       re.I)
# Acepta combinación (win+r) y tecla única (win) — el modelo a veces usa una sola.
_RE_HOTKEY       = re.compile(r"^hotkey\s+([\w]+(?:\+[\w]+)*)\s*$",      re.I)
_RE_WAIT         = re.compile(r"^wait\s+(\d+(?:\.\d+)?)\s*$",            re.I)
_RE_LAUNCH       = re.compile(r"^launch_app\s+(.+?)\s*$",                re.I)
# scroll up/down [N] — N opcional (por defecto _SCROLL_DEF). Acepta arriba/abajo.
_RE_SCROLL       = re.compile(r"^scroll\s+(up|down|arriba|abajo)(?:\s+(\d+))?\s*$", re.I)
_RE_DONE         = re.compile(r"^done\s*$",                              re.I)


class Controller:
    """Despachador de acciones físicas. `ejecutar()` retorna True/False."""

    def __init__(self) -> None:
        self.simulacion = not _OK
        modo = "SIMULACIÓN" if self.simulacion else "REAL"
        logger.info("Controller inicializado — modo físico: %s.", modo)

    @property
    def es_done(self) -> bool:
        """Se actualiza tras `ejecutar`: True si la última acción fue 'done'."""
        return self._ultimo_done

    _ultimo_done = False

    # ── API pública ────────────────────────────────────────────────────────────
    def ejecutar(self, accion: str, captura: Optional[Captura] = None) -> bool:
        """
        Parsea y ejecuta una ACCION. Las coordenadas se reescalan al espacio real
        usando `captura` (si se proporciona). Devuelve True si se ejecutó (o simuló)
        correctamente, False si el comando es inválido o falló.
        """
        self._ultimo_done = False
        cmd = (accion or "").strip()
        if not cmd:
            logger.warning("Controller: acción vacía.")
            return False

        try:
            m = _RE_DONE.match(cmd)
            if m:
                self._ultimo_done = True
                logger.info("Controller: 'done' — tarea señalada como completa.")
                return True

            m = _RE_LAUNCH.match(cmd)
            if m:
                return self._launch_app(m.group(1))

            m = _RE_CLICK.match(cmd)
            if m:
                return self._click(int(m.group(1)), int(m.group(2)), captura, doble=False)

            m = _RE_DOUBLE_CLICK.match(cmd)
            if m:
                return self._click(int(m.group(1)), int(m.group(2)), captura, doble=True)

            m = _RE_TYPE.match(cmd)
            if m:
                return self._type(m.group(1))

            m = _RE_KEY.match(cmd)
            if m:
                return self._key(m.group(1))

            m = _RE_HOTKEY.match(cmd)
            if m:
                teclas = [t.strip() for t in m.group(1).split("+") if t.strip()]
                return self._hotkey(teclas)

            m = _RE_SCROLL.match(cmd)
            if m:
                n = int(m.group(2)) if m.group(2) else _SCROLL_DEF
                return self._scroll(m.group(1), n)

            m = _RE_WAIT.match(cmd)
            if m:
                return self._wait(float(m.group(1)))

        except Exception as exc:                   # noqa: BLE001
            if "FailSafe" in type(exc).__name__:
                logger.warning("Controller: FailSafe de pyautogui activado.")
                return False
            logger.error("Controller: error ejecutando '%s': %s", cmd[:60], exc)
            return False

        logger.warning("Controller: comando no reconocido → '%s'", cmd[:80])
        return False

    # ── Apps ────────────────────────────────────────────────────────────────────
    def _launch_app(self, nombre: str) -> bool:
        """Lanza una app por nombre vía ShellExecute (os.startfile): resuelve los
        App Paths de Windows (notepad, calc, mspaint, explorer, msedge, chrome…).
        UNA acción atómica, sin simular win+r. Falla limpia si no se resuelve."""
        nombre = nombre.strip().strip('"').strip("'")
        if not nombre:
            return False
        if self.simulacion:
            logger.info("[SIM] launch_app('%s')", nombre)
            return True
        try:
            os.startfile(nombre)               # noqa: S606 (Windows ShellExecute)
        except (OSError, ValueError, AttributeError) as exc:
            logger.warning("launch_app: no se pudo lanzar '%s' — %s", nombre, exc)
            return False
        logger.info("launch_app '%s'.", nombre)
        return True

    # ── Mouse ───────────────────────────────────────────────────────────────────
    def _click(self, x_img: int, y_img: int, cap: Optional[Captura], doble: bool) -> bool:
        x, y = (cap.real(x_img, y_img) if cap else (x_img, y_img))
        etiqueta = "double_click" if doble else "click"
        if self._en_zona_esquina(x, y):
            logger.warning(
                "Controller: coordenadas en zona prohibida (esquina), modelo debe "
                "recalcular — %s real(%d,%d) [img(%d,%d)].",
                etiqueta, x, y, x_img, y_img,
            )
            return False
        if self.simulacion:
            logger.info("[SIM] %s img(%d,%d) → real(%d,%d)", etiqueta, x_img, y_img, x, y)
            return True
        if not self._mover_verificado(x, y, etiqueta):
            return False
        if doble:
            pyautogui.doubleClick()
        else:
            pyautogui.click()
        logger.info("%s en real(%d,%d) [img(%d,%d)].", etiqueta, x, y, x_img, y_img)
        return True

    def _tamano_pantalla(self) -> Optional[tuple[int, int]]:
        """Tamaño real de la pantalla (ancho, alto), o None si no se puede medir."""
        if pyautogui is None:
            return None
        try:
            ancho, alto = pyautogui.size()
            return int(ancho), int(alto)
        except Exception:                          # noqa: BLE001
            return None

    def _en_zona_esquina(self, x: int, y: int) -> bool:
        """True si (x, y) cae dentro de _MARGEN_ESQUINA de alguna de las 4 esquinas
        de la pantalla (la zona que dispara el FailSafe). Si no se puede medir la
        pantalla (modo SIMULACIÓN / sin pyautogui), no bloquea (devuelve False)."""
        tam = self._tamano_pantalla()
        if tam is None:
            return False
        ancho, alto = tam
        cerca_x = x <= _MARGEN_ESQUINA or x >= ancho - 1 - _MARGEN_ESQUINA
        cerca_y = y <= _MARGEN_ESQUINA or y >= alto - 1 - _MARGEN_ESQUINA
        return cerca_x and cerca_y

    def _mover_verificado(self, x: int, y: int, etiqueta: str) -> bool:
        """Mueve VISIBLEMENTE a (x, y) y confirma con el cursor medido; re-ancla
        ante interferencia transitoria (mano del usuario). No clica."""
        pyautogui.moveTo(x, y, duration=_DUR_MOV)
        for espera in (0.0, _ESPERA_REINT):
            px, py = pyautogui.position()
            if abs(px - x) <= _TOL_PX and abs(py - y) <= _TOL_PX:
                return True
            if espera:
                time.sleep(espera)
            pyautogui.moveTo(x, y)
        px, py = pyautogui.position()
        if abs(px - x) <= _TOL_PX and abs(py - y) <= _TOL_PX:
            return True
        logger.warning("mouse.%s ABORTADO → pedido (%d,%d), cursor en (%d,%d).",
                       etiqueta, x, y, px, py)
        return False

    def _scroll(self, direccion: str, n: int) -> bool:
        """Desplaza N clics de rueda. up/arriba → positivo; down/abajo → negativo.
        N se acota a [1, _SCROLL_MAX] (el modelo a veces pide cantidades enormes)."""
        n = min(max(1, n), _SCROLL_MAX)
        arriba = direccion.lower() in ("up", "arriba")
        clicks = n * _SCROLL_PASO * (1 if arriba else -1)
        if self.simulacion:
            logger.info("[SIM] scroll %s %d", direccion.lower(), n)
            return True
        pyautogui.scroll(clicks)
        logger.info("scroll %s %d (%d unidades).", direccion.lower(), n, clicks)
        return True

    # ── Teclado ──────────────────────────────────────────────────────────────────
    def _type(self, texto: str) -> bool:
        if self.simulacion:
            logger.info("[SIM] type('%s')", texto[:80])
            return True
        pyautogui.write(texto, interval=0.02)
        logger.info("type (%d chars).", len(texto))
        return True

    def _key(self, tecla: str) -> bool:
        if self.simulacion:
            logger.info("[SIM] key('%s')", tecla)
            return True
        pyautogui.press(tecla.lower())
        logger.info("key '%s'.", tecla)
        return True

    def _hotkey(self, teclas: list[str]) -> bool:
        if not teclas:
            return False
        norm = [t.lower() for t in teclas]
        if _es_hotkey_prohibida(norm):
            logger.warning("hotkey prohibida bloqueada: %s", "+".join(norm))
            return False
        if self.simulacion:
            logger.info("[SIM] hotkey(%s)", "+".join(norm))
            return True
        pyautogui.hotkey(*norm)
        # Asentamiento de diálogo (win+r y similares tardan ms en aparecer).
        settle = _COMBOS_DIALOGO.get(frozenset(norm))
        if settle:
            time.sleep(settle)
        logger.info("hotkey %s.", "+".join(norm))
        return True

    def _wait(self, segundos: float) -> bool:
        secs = min(max(0.0, segundos), _WAIT_MAX)
        logger.info("wait %.2fs.", secs)
        time.sleep(secs)
        return True
