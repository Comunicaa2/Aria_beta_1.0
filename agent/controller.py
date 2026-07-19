"""
agent/controller.py — Capa de automatización (mouse + teclado) de Aria 1.0.

Recibe la cadena de ACCION cruda que decidió Gemini y la ejecuta en el SO.
Comandos válidos (uno por turno):

    click X Y           clic izquierdo en (X, Y) del espacio IMAGEN
    double_click X Y    doble clic en (X, Y) del espacio IMAGEN
    right_click X Y     clic derecho en (X, Y) (menú contextual)
    drag X1 Y1 X2 Y2    arrastra desde (X1,Y1) hasta (X2,Y2) (mover/seleccionar)
    middle_click X Y    clic central en (X, Y)
    hover X Y           mueve el cursor a (X, Y) sin clicar (menús al pasar)
    hscroll DIR N       scroll horizontal N clics (hscroll right 5 / hscroll left 3)
    type TEXTO          escribe el texto exacto
    key TECLA           pulsa una tecla       (key enter / key esc / key tab)
    hotkey A+B[+C]      combinación           (hotkey ctrl+c / hotkey ctrl+s)
    hold_key MOD+click X Y  mantén MOD (shift/ctrl/alt) y clic en (X,Y) (multi-selección)
    hold_key MOD+key T      mantén MOD y pulsa la tecla T
    launch_app NOMBRE   lanza una app por nombre (launch_app notepad / calc / msedge)
    click_ui "T"        UI Automation: clica el elemento con nombre T en la ventana activa
    find_text "T"       OCR: localiza el texto T y reporta sus coords (no clica)
    find_image RUTA     localiza un PNG en pantalla y reporta sus coords (no clica)
    focus_window "T"    trae al frente la ventana cuyo título contiene T
    wait N              espera N segundos
    guardar NOMBRE      escribe el bloque CONTENIDO en workspace/NOMBRE (informes, scripts)
    ejecutar_python F   ejecuta workspace/F.py y reporta su salida (análisis de datos real)
    done                señala tarea completada (no toca el SO)

Las coordenadas que da el modelo están en el espacio de la IMAGEN reducida; este
módulo las reescala al espacio REAL de la pantalla usando la `Captura`.

Envuelve pyautogui de forma resiliente: si no está disponible, entra en modo
SIMULACIÓN (registra la acción pero no la ejecuta) y nunca lanza hacia arriba.
"""

import difflib
import logging
import os
import re
import subprocess
import sys
import time
from typing import Optional

from config import TIMEOUT_SCRIPT, WORKSPACE_DIR
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


# Control types de UI Automation sobre los que tiene sentido clicar por nombre.
# Excluye deliberadamente Text y contenedores (Pane, Group, Document…): eran la
# causa del misclick "no" → "Novedades"/párrafos largos.
_CONTROL_TYPES_ACCIONABLES = frozenset({
    "Button", "SplitButton", "CheckBox", "RadioButton", "ComboBox",
    "MenuItem", "ListItem", "DataItem", "TabItem", "TreeItem", "Hyperlink",
})


def _es_control_accionable(elem) -> bool:
    """True si el control type del elemento está en la allowlist accionable."""
    try:
        return elem.element_info.control_type in _CONTROL_TYPES_ACCIONABLES
    except Exception:                              # noqa: BLE001
        return False


def _buscar_por_palabra(obj: str, candidatos):
    """Busca `obj` como PALABRA COMPLETA en los nombres de `candidatos`
    [(nombre, elem)]. Devuelve (elem, n_matches). Con varios matches desambigua
    por similitud (difflib) y, en empate, por nombre más corto; empate genuino
    (misma similitud Y longitud) → (None, n) con n>=2: mejor fallar que adivinar."""
    patron = re.compile(rf"\b{re.escape(obj)}\b", re.I)
    matches = [(n, e) for n, e in candidatos if patron.search(n)]
    if not matches:
        return None, 0
    if len(matches) == 1:
        return matches[0][1], 1
    puntuados = sorted(
        ((difflib.SequenceMatcher(None, obj, n.lower()).ratio(), -len(n), n, e)
         for n, e in matches),
        key=lambda t: (t[0], t[1]), reverse=True,
    )
    mejor, segundo = puntuados[0], puntuados[1]
    if mejor[0] == segundo[0] and len(mejor[2]) == len(segundo[2]):
        return None, len(matches)
    return mejor[3], len(matches)


# ─── Patrones de comandos ─────────────────────────────────────────────────────
_RE_CLICK        = re.compile(r"^click\s+(-?\d+)\s+(-?\d+)\s*$",        re.I)
_RE_DOUBLE_CLICK = re.compile(r"^double_click\s+(-?\d+)\s+(-?\d+)\s*$", re.I)
_RE_RIGHT_CLICK  = re.compile(r"^right_click\s+(-?\d+)\s+(-?\d+)\s*$",   re.I)
_RE_DRAG         = re.compile(r"^drag\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*$", re.I)
_RE_MIDDLE_CLICK = re.compile(r"^middle_click\s+(-?\d+)\s+(-?\d+)\s*$", re.I)
_RE_HOVER        = re.compile(r"^hover\s+(-?\d+)\s+(-?\d+)\s*$",        re.I)
_RE_HSCROLL      = re.compile(r"^hscroll\s+(left|right|izquierda|derecha)(?:\s+(\d+))?\s*$", re.I)
_RE_HOLD_KEY     = re.compile(r"^hold_key\s+(.+?)\+(click|key)\s+(.+?)\s*$", re.I)
_RE_TYPE         = re.compile(r"^type\s+(.+)$",                          re.I | re.S)
_RE_KEY          = re.compile(r"^key\s+(\S+)\s*$",                       re.I)
# Acepta combinación (win+r) y tecla única (win) — el modelo a veces usa una sola.
_RE_HOTKEY       = re.compile(r"^hotkey\s+([\w]+(?:\+[\w]+)*)\s*$",      re.I)
_RE_WAIT         = re.compile(r"^wait\s+(\d+(?:\.\d+)?)\s*$",            re.I)
_RE_LAUNCH       = re.compile(r"^launch_app\s+(.+?)\s*$",                re.I)
_RE_CLICK_UI     = re.compile(r'^click_ui\s+["\']?(.+?)["\']?\s*$',      re.I)
_RE_FIND_TEXT    = re.compile(r'^find_text\s+["\']?(.+?)["\']?\s*$',     re.I)
_RE_FIND_IMAGE   = re.compile(r'^find_image\s+["\']?(.+?)["\']?\s*$',    re.I)
_RE_FOCUS_WIN    = re.compile(r'^focus_window\s+["\']?(.+?)["\']?\s*$',  re.I)
# scroll up/down [N] — N opcional (por defecto _SCROLL_DEF). Acepta arriba/abajo.
_RE_SCROLL       = re.compile(r"^scroll\s+(up|down|arriba|abajo)(?:\s+(\d+))?\s*$", re.I)
_RE_DONE         = re.compile(r"^done\s*$",                              re.I)
# Nombre de archivo SIMPLE (empieza por alfanumérico; sin rutas, sin '..', sin
# barras): es la frontera de confianza de guardar/ejecutar_python — todo vive
# dentro de WORKSPACE_DIR.
_RE_GUARDAR      = re.compile(r'^guardar\s+["\']?([\w][\w.\- ]*)["\']?\s*$', re.I)
_RE_EJEC_PY      = re.compile(r'^ejecutar_python\s+["\']?([\w][\w.\- ]*\.py)["\']?\s*$', re.I)


class _CapturaDetalle(logging.Handler):
    """FIX #8: guarda el último WARNING/ERROR del controller en `ultimo_detalle`,
    para que las razones de fallo (esquina, comando no reconocido, app no lanzada,
    hotkey prohibida, movimiento abortado…) lleguen a Gemini en la nota de
    resultado. Los éxitos loguean en INFO, así que no se capturan."""

    def __init__(self, ctrl: "Controller") -> None:
        super().__init__(level=logging.WARNING)
        self._ctrl = ctrl

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._ctrl.ultimo_detalle = record.getMessage()
        except Exception:                          # noqa: BLE001
            pass


class Controller:
    """Despachador de acciones físicas. `ejecutar()` retorna True/False."""

    def __init__(self) -> None:
        self.simulacion = not _OK
        modo = "SIMULACIÓN" if self.simulacion else "REAL"
        logger.info("Controller inicializado — modo físico: %s.", modo)
        # FIX #8: captura el último WARNING del controller en ultimo_detalle, para
        # que las razones de fallo lleguen a Gemini. Evita duplicar el handler si
        # se crean varios Controller (p. ej. en tests).
        logger.handlers = [h for h in logger.handlers if not isinstance(h, _CapturaDetalle)]
        logger.addHandler(_CapturaDetalle(self))

    @property
    def es_done(self) -> bool:
        """Se actualiza tras `ejecutar`: True si la última acción fue 'done'."""
        return self._ultimo_done

    _ultimo_done = False
    ultimo_detalle = ""   # detalle de la última acción (p. ej. coords de find_text/find_image)

    # ── API pública ────────────────────────────────────────────────────────────
    def ejecutar(self, accion: str, captura: Optional[Captura] = None,
                 contenido: str = "") -> bool:
        """
        Parsea y ejecuta una ACCION. Las coordenadas se reescalan al espacio real
        usando `captura` (si se proporciona). `contenido` es el bloque CONTENIDO
        multilínea de la acción 'guardar'. Devuelve True si se ejecutó (o simuló)
        correctamente, False si el comando es inválido o falló.
        """
        self._ultimo_done = False
        self.ultimo_detalle = ""
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

            m = _RE_CLICK_UI.match(cmd)
            if m:
                return self._click_ui(m.group(1))

            m = _RE_FIND_TEXT.match(cmd)
            if m:
                return self._find_text(m.group(1), captura)

            m = _RE_FIND_IMAGE.match(cmd)
            if m:
                return self._find_image(m.group(1), captura)

            m = _RE_FOCUS_WIN.match(cmd)
            if m:
                return self._focus_window(m.group(1))

            m = _RE_CLICK.match(cmd)
            if m:
                return self._click(int(m.group(1)), int(m.group(2)), captura, doble=False)

            m = _RE_DOUBLE_CLICK.match(cmd)
            if m:
                return self._click(int(m.group(1)), int(m.group(2)), captura, doble=True)

            m = _RE_RIGHT_CLICK.match(cmd)
            if m:
                return self._click_boton(int(m.group(1)), int(m.group(2)), captura, "right")

            m = _RE_DRAG.match(cmd)
            if m:
                return self._drag(int(m.group(1)), int(m.group(2)),
                                  int(m.group(3)), int(m.group(4)), captura)

            m = _RE_TYPE.match(cmd)
            if m:
                if not self._ventana_activa_valida():
                    logger.warning("type sin ventana activa que reciba el input "
                                   "(foco en escritorio o consola).")
                    return False
                return self._type(m.group(1))

            m = _RE_KEY.match(cmd)
            if m:
                if not self._ventana_activa_valida():
                    logger.warning("key sin ventana activa que reciba el input "
                                   "(foco en escritorio o consola).")
                    return False
                return self._key(m.group(1))

            m = _RE_HOTKEY.match(cmd)
            if m:
                teclas = [t.strip() for t in m.group(1).split("+") if t.strip()]
                return self._hotkey(teclas)

            m = _RE_HOLD_KEY.match(cmd)
            if m:
                return self._hold_key(m.group(1), m.group(2).lower(), m.group(3), captura)

            m = _RE_SCROLL.match(cmd)
            if m:
                n = int(m.group(2)) if m.group(2) else _SCROLL_DEF
                return self._scroll(m.group(1), n)

            m = _RE_MIDDLE_CLICK.match(cmd)
            if m:
                return self._click_boton(int(m.group(1)), int(m.group(2)), captura, "middle")

            m = _RE_HOVER.match(cmd)
            if m:
                return self._hover(int(m.group(1)), int(m.group(2)), captura)

            m = _RE_HSCROLL.match(cmd)
            if m:
                n = int(m.group(2)) if m.group(2) else _SCROLL_DEF
                return self._hscroll(m.group(1), n)

            m = _RE_WAIT.match(cmd)
            if m:
                return self._wait(float(m.group(1)))

            m = _RE_GUARDAR.match(cmd)
            if m:
                return self._guardar(m.group(1).strip(), contenido)

            m = _RE_EJEC_PY.match(cmd)
            if m:
                return self._ejecutar_python(m.group(1).strip())

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

    # ── Trabajo con archivos (workspace) ────────────────────────────────────────
    def _guardar(self, nombre: str, contenido: str) -> bool:
        """Escribe `contenido` en WORKSPACE_DIR/nombre. El regex ya restringe el
        nombre a un archivo simple; el realpath-check es el cinturón extra de la
        frontera de confianza (el nombre lo decide el modelo)."""
        if not contenido.strip():
            logger.warning("guardar '%s': falta el bloque CONTENIDO con el "
                           "contenido del archivo.", nombre)
            return False
        os.makedirs(WORKSPACE_DIR, exist_ok=True)
        ruta = os.path.realpath(os.path.join(WORKSPACE_DIR, nombre))
        if os.path.dirname(ruta) != os.path.realpath(WORKSPACE_DIR):
            logger.warning("guardar: nombre inválido '%s' (solo archivos simples, "
                           "sin rutas).", nombre)
            return False
        try:
            with open(ruta, "w", encoding="utf-8", newline="\n") as f:
                f.write(contenido if contenido.endswith("\n") else contenido + "\n")
        except OSError as exc:
            logger.warning("guardar '%s': %s", nombre, exc)
            return False
        logger.info("guardar '%s' (%d caracteres).", nombre, len(contenido))
        self.ultimo_detalle = (f"Archivo '{nombre}' guardado en la carpeta de "
                               f"trabajo ({len(contenido)} caracteres)")
        return True

    def _ejecutar_python(self, nombre: str) -> bool:
        """Ejecuta WORKSPACE_DIR/nombre con el mismo intérprete de Aria y deja la
        salida (o el error) en `ultimo_detalle` para que el modelo la lea en el
        siguiente turno. Timeout defensivo: TIMEOUT_SCRIPT."""
        ruta = os.path.realpath(os.path.join(WORKSPACE_DIR, nombre))
        if os.path.dirname(ruta) != os.path.realpath(WORKSPACE_DIR):
            logger.warning("ejecutar_python: nombre inválido '%s'.", nombre)
            return False
        if not os.path.isfile(ruta):
            logger.warning("ejecutar_python: '%s' no existe en la carpeta de "
                           "trabajo — guárdalo primero con 'guardar'.", nombre)
            return False
        try:
            r = subprocess.run(
                [sys.executable, nombre], cwd=WORKSPACE_DIR,
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=TIMEOUT_SCRIPT,
            )
        except subprocess.TimeoutExpired:
            logger.warning("ejecutar_python: '%s' superó los %ds y fue cancelado.",
                           nombre, TIMEOUT_SCRIPT)
            return False
        except OSError as exc:
            logger.warning("ejecutar_python '%s': %s", nombre, exc)
            return False
        # ponytail: solo los últimos 700 chars viajan al modelo; suficiente para
        # métricas impresas — subir el tope si algún análisis lo necesita.
        salida = (r.stdout or "").strip()[-700:]
        if r.returncode != 0:
            err = ((r.stderr or "").strip() or salida or "sin salida")[-700:]
            logger.warning("ejecutar_python '%s' terminó con error (rc=%d): %s",
                           nombre, r.returncode, err)
            return False
        logger.info("ejecutar_python '%s' OK.", nombre)
        self.ultimo_detalle = f"Salida de {nombre}: {salida or '(sin salida)'}"
        return True

    def _click_ui(self, texto: str) -> bool:
        """Clica el elemento de UI con NOMBRE (accesibilidad) `texto`, buscándolo
        con Windows UI Automation en la ventana en PRIMER PLANO. Orden: exacta →
        palabra completa entre controles ACCIONABLES (botones, menús…) → palabra
        completa sin filtro si el árbol no expone tipos (Electron/web). Ambiguo o
        sin match → falla limpio (jamás clica a ciegas por substring). El clic
        reusa el camino verificado de _click (corner-check + cursor medido).
        Degrada limpio si pywinauto no está instalado."""
        # ponytail: busca SOLO la ventana activa; ampliar a todas si hace falta.
        texto = texto.strip().strip('"').strip("'")
        if not texto:
            return False
        if self.simulacion:
            logger.info("[SIM] click_ui('%s')", texto)
            return True
        try:
            from pywinauto import Desktop
        except ImportError as exc:
            logger.warning("click_ui: pywinauto no disponible (%s).", exc)
            self.ultimo_detalle = "click_ui: pywinauto no instalado"
            return False
        try:
            import ctypes
            fg = ctypes.windll.user32.GetForegroundWindow()
            if not fg:
                self.ultimo_detalle = "click_ui: no hay ventana activa donde buscar"
                return False
            ventana = Desktop(backend="uia").window(handle=fg).wrapper_object()
            candidatos = [(e.window_text(), e) for e in ventana.descendants()
                          if e.window_text()]
        except Exception as exc:                   # noqa: BLE001
            logger.warning("click_ui: fallo consultando UI Automation — %s", exc)
            self.ultimo_detalle = "click_ui: fallo leyendo el árbol de UI"
            return False

        obj = texto.lower()
        elem = next((e for n, e in candidatos if n.lower() == obj), None)
        if elem is None:
            # Sin exacta: palabra completa SOLO entre controles accionables
            # (evita matchear "no" contra "Novedades" o párrafos de texto).
            accionables = [(n, e) for n, e in candidatos if _es_control_accionable(e)]
            elem, n_matches = _buscar_por_palabra(obj, accionables)
            if elem is None and n_matches == 0 and not accionables:
                # Árbol UIA sin control types útiles (típico en Electron/web):
                # reintento sin filtro de tipo, pero siempre por palabra completa.
                logger.warning("click_ui: sin controles accionables en el árbol — "
                               "reintento sin filtro de tipo.")
                elem, n_matches = _buscar_por_palabra(obj, candidatos)
            if elem is None:
                if n_matches >= 2:
                    self.ultimo_detalle = (f"click_ui: '{texto}' es ambiguo "
                                           f"({n_matches} elementos coinciden) — "
                                           "usa el nombre completo o find_text")
                else:
                    self.ultimo_detalle = (f"click_ui: elemento '{texto}' no existe en la "
                                           "ventana activa — prueba find_text o click X Y")
                logger.info("click_ui: '%s' sin match claro (%d candidatos, %d coincidencias).",
                            texto, len(candidatos), n_matches)
                return False
        r = elem.rectangle()
        x, y = r.mid_point().x, r.mid_point().y
        nombre = elem.window_text()[:40]
        if not self._click(x, y, cap=None, doble=False):   # coords ya REALES
            return False
        self.ultimo_detalle = f"click_ui: '{nombre}' clicado en ({x},{y})"
        logger.info("click_ui: '%s' → real(%d,%d).", nombre, x, y)
        return True

    def _find_text(self, objetivo: str, cap: Optional[Captura]) -> bool:
        """OCR sobre la imagen que ve el modelo (cap.b64). Si encuentra `objetivo`
        (case-insensitive), reporta sus coords en ESPACIO IMAGEN vía ultimo_detalle
        para que el modelo haga 'click X Y'. NO clica ni mueve el ratón (lectura
        pura). Degrada limpio si Tesseract no está instalado."""
        objetivo = objetivo.strip().strip('"').strip("'")
        if not objetivo:
            return False
        if cap is None or not cap.b64:
            self.ultimo_detalle = "find_text: sin captura para analizar"
            return False
        try:
            import base64, io
            from PIL import Image
            import pytesseract
            from pytesseract import Output
        except ImportError as exc:
            logger.warning("find_text: OCR no disponible (%s).", exc)
            self.ultimo_detalle = "find_text: OCR no instalado"
            return False
        # Si TESSERACT_CMD apunta a un exe existente, úsalo (cuando tesseract no
        # está en el PATH del proceso). Robusto entre máquinas; lo carga el .env.
        cmd = os.getenv("TESSERACT_CMD", "").strip()
        if cmd and os.path.isfile(cmd):
            pytesseract.pytesseract.tesseract_cmd = cmd
        try:
            img = Image.open(io.BytesIO(base64.b64decode(cap.b64)))
            datos = pytesseract.image_to_data(img, output_type=Output.DICT)
        except pytesseract.TesseractNotFoundError:
            logger.warning("find_text: binario Tesseract no encontrado en PATH.")
            self.ultimo_detalle = "find_text: Tesseract no instalado"
            return False
        except Exception as exc:                   # noqa: BLE001
            logger.warning("find_text: fallo OCR — %s", exc)
            self.ultimo_detalle = "find_text: fallo OCR"
            return False

        obj = objetivo.lower()
        n = len(datos["text"])
        # 1) palabra única que contiene el objetivo
        for i in range(n):
            palabra = (datos["text"][i] or "").strip()
            if palabra and obj in palabra.lower():
                cx = datos["left"][i] + datos["width"][i] // 2
                cy = datos["top"][i] + datos["height"][i] // 2
                self.ultimo_detalle = (f"texto '{objetivo}' encontrado en "
                                       f"({cx},{cy}) — usa click {cx} {cy}")
                logger.info("find_text: '%s' en img(%d,%d).", objetivo, cx, cy)
                return True
        # 2) objetivo multi-palabra: agrupar por línea y unir cajas
        lineas: dict = {}
        for i in range(n):
            if (datos["text"][i] or "").strip():
                clave = (datos["block_num"][i], datos["par_num"][i], datos["line_num"][i])
                lineas.setdefault(clave, []).append(i)
        for idxs in lineas.values():
            texto_linea = " ".join((datos["text"][i] or "").strip() for i in idxs)
            if obj in texto_linea.lower():
                x0 = min(datos["left"][i] for i in idxs)
                y0 = min(datos["top"][i] for i in idxs)
                x1 = max(datos["left"][i] + datos["width"][i] for i in idxs)
                y1 = max(datos["top"][i] + datos["height"][i] for i in idxs)
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                self.ultimo_detalle = (f"texto '{objetivo}' encontrado en "
                                       f"({cx},{cy}) — usa click {cx} {cy}")
                logger.info("find_text: '%s' (línea) en img(%d,%d).", objetivo, cx, cy)
                return True
        self.ultimo_detalle = f"texto '{objetivo}' no encontrado en pantalla"
        logger.info("find_text: '%s' no encontrado.", objetivo)
        return False

    def _find_image(self, ruta: str, cap: Optional[Captura]) -> bool:
        """Localiza una plantilla PNG en pantalla (pyautogui.locateCenterOnScreen,
        confidence 0.8 vía OpenCV). locate da coords REALES → se convierten a
        ESPACIO IMAGEN (inverso de cap.real) y se reportan vía ultimo_detalle para
        que el modelo haga 'click X Y'. NO clica (lectura pura). Falla limpia."""
        ruta = ruta.strip().strip('"').strip("'")
        if not ruta:
            return False
        if not os.path.isfile(ruta):
            self.ultimo_detalle = f"find_image: archivo no encontrado '{ruta}'"
            logger.warning("find_image: no existe el archivo '%s'.", ruta)
            return False
        if pyautogui is None:
            self.ultimo_detalle = "find_image: pyautogui no disponible"
            return False
        try:
            centro = pyautogui.locateCenterOnScreen(ruta, confidence=0.8)
        except Exception as exc:                   # ImageNotFound / OSError / cv2
            logger.info("find_image: '%s' no localizada (%s).", ruta, type(exc).__name__)
            centro = None
        if centro is None:
            self.ultimo_detalle = f"imagen '{os.path.basename(ruta)}' no encontrada en pantalla"
            return False
        x_real, y_real = int(centro[0]), int(centro[1])
        if cap is not None and cap.ancho_img and cap.alto_img:
            x_img = int(round(x_real / cap.escala_x))
            y_img = int(round(y_real / cap.escala_y))
        else:
            x_img, y_img = x_real, y_real          # sin captura: no se puede reescalar
        self.ultimo_detalle = (f"imagen '{os.path.basename(ruta)}' encontrada en "
                               f"({x_img},{y_img}) — usa click {x_img} {y_img}")
        logger.info("find_image: '%s' en img(%d,%d) [real(%d,%d)].",
                    ruta, x_img, y_img, x_real, y_real)
        return True

    def _focus_window(self, titulo: str) -> bool:
        """Trae al frente la ventana cuyo título CONTIENE `titulo` (case-insensitive).
        Si está minimizada, la restaura. Best-effort: en Windows el 'foreground lock'
        puede impedir el cambio; el modelo lo verifica en la siguiente captura."""
        titulo = titulo.strip().strip('"').strip("'")
        if not titulo:
            return False
        if self.simulacion:
            logger.info("[SIM] focus_window('%s')", titulo)
            return True
        try:
            import pygetwindow as gw
        except ImportError as exc:
            logger.warning("focus_window: pygetwindow no disponible (%s).", exc)
            return False
        try:
            objetivo = titulo.lower()
            coincidentes = [w for w in gw.getAllWindows()
                            if w.title and objetivo in w.title.lower()]
        except Exception as exc:                   # noqa: BLE001
            logger.warning("focus_window: fallo buscando '%s' — %s", titulo, exc)
            return False
        if not coincidentes:
            self.ultimo_detalle = f"focus_window: ventana '{titulo}' no encontrada"
            logger.info("focus_window: '%s' no encontrada.", titulo)
            return False
        win = coincidentes[0]
        try:
            if win.isMinimized:
                win.restore()
            win.activate()
        except Exception as exc:                   # pygetwindow lanza incluso en éxito
            logger.info("focus_window: activate('%s') best-effort (%s).",
                        win.title, type(exc).__name__)
        logger.info("focus_window '%s'.", win.title)
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

    def _click_boton(self, x_img: int, y_img: int, cap: Optional[Captura], boton: str) -> bool:
        """Clic con botón no-izquierdo ('right'/'middle') en (X, Y) del espacio
        IMAGEN. Mismo patrón que _click: corner-check (FIX 1) + movimiento
        verificado. No modifica _click (las 7 core quedan intactas)."""
        x, y = (cap.real(x_img, y_img) if cap else (x_img, y_img))
        if self._en_zona_esquina(x, y):
            logger.warning("Controller: coordenadas en zona prohibida (esquina), modelo "
                           "debe recalcular — %s_click real(%d,%d) [img(%d,%d)].",
                           boton, x, y, x_img, y_img)
            return False
        if self.simulacion:
            logger.info("[SIM] %s_click img(%d,%d) → real(%d,%d)", boton, x_img, y_img, x, y)
            return True
        if not self._mover_verificado(x, y, f"{boton}_click"):
            return False
        pyautogui.click(button=boton)
        logger.info("%s_click en real(%d,%d) [img(%d,%d)].", boton, x, y, x_img, y_img)
        return True

    def _drag(self, x1_img: int, y1_img: int, x2_img: int, y2_img: int,
              cap: Optional[Captura]) -> bool:
        """Arrastra de (X1,Y1) a (X2,Y2) en espacio IMAGEN. Acción compuesta
        PERMITIDA (moveTo verificado al inicio + dragTo al destino). Corner-check
        en AMBOS extremos (FIX 1)."""
        x1, y1 = (cap.real(x1_img, y1_img) if cap else (x1_img, y1_img))
        x2, y2 = (cap.real(x2_img, y2_img) if cap else (x2_img, y2_img))
        if self._en_zona_esquina(x1, y1) or self._en_zona_esquina(x2, y2):
            logger.warning("Controller: coordenadas en zona prohibida (esquina), modelo "
                           "debe recalcular — drag real(%d,%d)->(%d,%d).", x1, y1, x2, y2)
            return False
        if self.simulacion:
            logger.info("[SIM] drag real(%d,%d) -> (%d,%d)", x1, y1, x2, y2)
            return True
        if not self._mover_verificado(x1, y1, "drag-inicio"):
            return False
        pyautogui.dragTo(x2, y2, duration=_DUR_MOV)
        logger.info("drag real(%d,%d) -> (%d,%d).", x1, y1, x2, y2)
        return True

    def _hover(self, x_img: int, y_img: int, cap: Optional[Captura]) -> bool:
        """Mueve el cursor a (X, Y) del espacio IMAGEN SIN clicar (menús que
        aparecen al pasar el ratón). Corner-check (FIX 1) + movimiento verificado."""
        x, y = (cap.real(x_img, y_img) if cap else (x_img, y_img))
        if self._en_zona_esquina(x, y):
            logger.warning("Controller: coordenadas en zona prohibida (esquina), modelo "
                           "debe recalcular — hover real(%d,%d) [img(%d,%d)].",
                           x, y, x_img, y_img)
            return False
        if self.simulacion:
            logger.info("[SIM] hover img(%d,%d) → real(%d,%d)", x_img, y_img, x, y)
            return True
        if not self._mover_verificado(x, y, "hover"):
            return False
        logger.info("hover en real(%d,%d) [img(%d,%d)].", x, y, x_img, y_img)
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

    def _ventana_activa_valida(self) -> bool:
        """True si hay una ventana en primer plano que pueda RECIBIR input: no 0,
        no la consola de Aria, no el escritorio (Progman/WorkerW). Evita escribir
        'al vacío' (FIX #9). Ante error, no bloquea (degradación segura)."""
        try:
            import ctypes
            u32 = ctypes.windll.user32
            fg = u32.GetForegroundWindow()
            if not fg:
                return False
            if fg == ctypes.windll.kernel32.GetConsoleWindow():
                return False
            # Bajo Windows Terminal la consola real es la ventana del terminal
            # ancestro, no la de GetConsoleWindow (oculta por ConPTY).
            from utils.image import hwnd_consola
            if fg == hwnd_consola():
                return False
            buf = ctypes.create_unicode_buffer(256)
            u32.GetClassNameW(fg, buf, 256)
            return buf.value not in ("Progman", "WorkerW")   # escritorio
        except Exception:                          # noqa: BLE001
            return True

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

    def _hscroll(self, direccion: str, n: int) -> bool:
        """Scroll HORIZONTAL de N clics. right/derecha → positivo; left/izquierda →
        negativo. N se acota a [1, _SCROLL_MAX]. Espeja a _scroll (mismas constantes)."""
        n = min(max(1, n), _SCROLL_MAX)
        derecha = direccion.lower() in ("right", "derecha")
        clicks = n * _SCROLL_PASO * (1 if derecha else -1)
        if self.simulacion:
            logger.info("[SIM] hscroll %s %d", direccion.lower(), n)
            return True
        pyautogui.hscroll(clicks)
        logger.info("hscroll %s %d (%d unidades).", direccion.lower(), n, clicks)
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

    def _hold_key(self, mods_str: str, accion: str, arg: str,
                  cap: Optional[Captura]) -> bool:
        """Mantén uno o más MODIFICADORES (shift/ctrl/alt/win) mientras se hace UNA
        acción: click X Y o key T. Excepción consciente al 'no macros' usando el
        idiom nativo pyautogui.hold(). El form +key se valida contra la lista negra
        (evita bypass tipo alt+key f4). Corner-check en el form +click."""
        permitidos = {"shift", "ctrl", "alt", "win"}
        mods = [_ALIAS_TECLAS.get(t.strip().lower(), t.strip().lower())
                for t in mods_str.split("+") if t.strip()]
        if not mods or any(m not in permitidos for m in mods):
            logger.warning("hold_key: modificadores inválidos %s (usa shift/ctrl/alt/win).", mods)
            return False

        if accion == "click":
            partes = arg.split()
            if len(partes) != 2:
                return False
            try:
                x_img, y_img = int(partes[0]), int(partes[1])
            except ValueError:
                return False
            x, y = (cap.real(x_img, y_img) if cap else (x_img, y_img))
            if self._en_zona_esquina(x, y):
                logger.warning("Controller: coordenadas en zona prohibida (esquina), modelo "
                               "debe recalcular — hold_key %s+click real(%d,%d).",
                               "+".join(mods), x, y)
                return False
            if self.simulacion:
                logger.info("[SIM] hold_key %s+click img(%d,%d) → real(%d,%d)",
                            "+".join(mods), x_img, y_img, x, y)
                return True
            if not self._mover_verificado(x, y, "hold_click"):
                return False
            with pyautogui.hold(mods):
                pyautogui.click()
            logger.info("hold_key %s+click en real(%d,%d).", "+".join(mods), x, y)
            return True

        # accion == "key"
        tecla = arg.strip().split()[0].lower() if arg.strip() else ""
        if not tecla:
            return False
        tecla = _ALIAS_TECLAS.get(tecla, tecla)
        if _es_hotkey_prohibida(mods + [tecla]):
            logger.warning("hold_key: combinación prohibida bloqueada: %s+%s",
                           "+".join(mods), tecla)
            return False
        if self.simulacion:
            logger.info("[SIM] hold_key %s+key %s", "+".join(mods), tecla)
            return True
        with pyautogui.hold(mods):
            pyautogui.press(tecla)
        logger.info("hold_key %s+key %s.", "+".join(mods), tecla)
        return True

    def _wait(self, segundos: float) -> bool:
        secs = min(max(0.0, segundos), _WAIT_MAX)
        logger.info("wait %.2fs.", secs)
        time.sleep(secs)
        return True
