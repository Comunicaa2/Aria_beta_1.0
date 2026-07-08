"""
core/brain.py — Cerebro de Aria 1.0 (Gemini 2.5 Flash, multimodal nativo).

Responsabilidades:
  1. Construir la petición a Gemini con IMAGEN + COMANDO en un solo flujo.
  2. Hacer cumplir el formato rígido de respuesta (PENSAMIENTO/ACCION/FIN).
  3. Mapear la profundidad de razonamiento al presupuesto de pensamiento de
     Gemini (THINKING → con budget; WORKING/OVERLOADED → sin budget, máx. velocidad).
  4. Gestionar el historial MÍNIMO y un recolector de basura visual: solo el
     último turno conserva la imagen (ahorro de tokens / latencia).
  5. Detectar el límite de API (HTTP 429) y propagarlo como `LimiteAPIError`
     para que el orquestador guarde el estado y se detenga limpiamente.

Sin relleno conversacional: la salida es siempre un comando estructurado.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from core import lecciones
from config import (
    AGENT_NAME,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_URL,
    GEN_MAX_TOKENS,
    GEN_STOP,
    GEN_TEMPERATURE,
    GEN_TOP_P,
    MAX_HISTORIAL,
    NVIDIA_API_KEY,
    NVIDIA_API_URL,
    NVIDIA_FALLBACK_MODELS,
    THINK_BUDGET_PROFUNDO,
    THINK_BUDGET_RAPIDO,
    THINK_RESPUESTA_EXTRA,
    TIMEOUT_CONNECT,
    TIMEOUT_POOL,
    TIMEOUT_READ,
    TIMEOUT_WRITE,
    TRAINING_MODE,
)

logger = logging.getLogger("aria.brain")

# Identificadores del modelo que produjo una acción (para los logs de Aria y el
# panel del Entrenador). Sus valores deben coincidir con trainer/protocolo.py.
MODELO_GEMINI    = "GEMINI"
MODELO_NIM_LLAMA = "NIM-LLAMA"   # meta/llama-3.2-90b-vision-instruct
MODELO_NIM_OMNI  = "NIM-MINIMAX"  # minimaxai/minimax-m3


def _etiqueta_nim(modelo: str) -> str:
    """Mapea el id de un modelo NIM a su etiqueta corta (robusto a overrides)."""
    return MODELO_NIM_LLAMA if "llama" in modelo.lower() else MODELO_NIM_OMNI

# ─── Rate limiter de Gemini ───────────────────────────────────────────────────
# Gemini 2.5 Flash (free tier) = 15 RPM = 1 req/4s. Forzamos un mínimo de 4.5 s
# entre llamadas para no chocar contra el límite (margen de seguridad sobre los 4s).
GEMINI_MIN_INTERVALO = 4.5

# ── Rate limiter CROSS-PROCESS (compartido con el entrenador) ─────────────────
# Aria y el entrenador comparten la cuota de 15 RPM de la MISMA API key. Se
# coordinan por un archivo de marca (tasks/rate_gemini.json) protegido con un
# mutex de exclusión mutua (lock O_EXCL): antes de cada llamada a Gemini se espera
# a que hayan pasado ≥ GEMINI_MIN_INTERVALO s desde la última llamada de CUALQUIER
# proceso. Usa time.time() (reloj de pared, comparable entre procesos). Si el
# archivo no está disponible, cae a un piso en memoria (no rompe el comportamiento).
_RATE_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks")
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


def _rate_limit_compartido() -> None:
    """
    Garantiza ≥ GEMINI_MIN_INTERVALO s entre llamadas a Gemini de CUALQUIER proceso
    (Aria + entrenador), vía el archivo de marca bajo mutex. Atómico: la sección
    leer→esperar→escribir corre dentro del lock. Degrada a piso en memoria si el
    sistema de archivos no está disponible (sin romper nada).
    """
    global _rate_ultima_local
    try:
        os.makedirs(_RATE_DIR, exist_ok=True)
    except OSError:
        pass

    if not _rate_lock_adquirir():                  # no se pudo coordinar → piso local
        logger.error("Rate limit: NO se pudo adquirir el mutex en %.0fs — degradando a "
                     "piso LOCAL; Aria y el entrenador podrían exceder %.0f RPM combinados.",
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

# Tokens máximos para el fallback NIM. El modelo razona en un campo aparte
# (reasoning_content) que comparte presupuesto con la respuesta: necesita margen
# amplio para no agotar el budget antes de emitir la respuesta en formato.
_NIM_MAX_TOKENS = 2048
# El timeout de UNA llamada NIM debe ser MENOR que el timeout de tarea del trainer
# (30s congelado en NIM, FIX #3): si NIM no responde en 25s, fallamos rápido en vez
# de bloquear la tarea hasta que el trainer la expire. (FIX #6)
_NIM_TIMEOUT = 25.0
# Limpia los bloques de razonamiento <think>…</think> que emiten los modelos de
# razonamiento, dejando solo la respuesta en formato PENSAMIENTO/ACCION.
_RE_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_RE_THINK_ABIERTO = re.compile(r"<think>.*\Z", re.S | re.I)

# Aislamiento agresivo del bloque de formato en la salida de NIM, que a veces viene
# envuelta en JSON/literales y deja basura (p. ej. "...ACCION: done'}]"). Extraemos
# SOLO PENSAMIENTO…ACCION y descartamos todo lo demás (antes y después).
# Las comillas opcionales (["']?) toleran que el modelo envuelva el bloque en
# JSON/diccionario: "PENSAMIENTO": "...", "ACCION": "click 1 2".
_RE_BLOQUE_FORMATO = re.compile(
    r"PENSAMIENTO\s*[\"']?\s*:\s*[\"']?(?P<pens>.*?)[\"']?\s*"
    r"ACCI[OÓ]N\s*[\"']?\s*:\s*[\"']?(?P<acc>.+?)\s*(?:\bFIN\b|[\r\n]|$)",
    re.I | re.S,
)
_RE_SOLO_ACCION = re.compile(
    r"ACCI[OÓ]N\s*[\"']?\s*:\s*[\"']?(?P<acc>.+?)\s*(?:\bFIN\b|[\r\n}\]]|$)", re.I | re.S
)


def _limpiar_think(texto: str) -> str:
    """Quita bloques <think> (cerrados o sin cerrar) de la salida de un modelo razonador."""
    if "<think>" not in texto.lower():
        return texto
    texto = _RE_THINK.sub("", texto)
    texto = _RE_THINK_ABIERTO.sub("", texto)   # <think> sin cierre (cortado) → fuera
    return texto.strip()


def _limpiar_accion(acc: str) -> str:
    """Recorta basura JSON/comillas del valor de ACCION (p. ej. "done'}]" → "done")."""
    acc = re.split(r"[\r\n}\]]", acc)[0]             # corta en el 1.er ruido estructural
    return acc.strip().strip("'\"").rstrip("'\"}],) ").strip()


def _aislar_formato(texto: str) -> str:
    """
    Limpieza AGRESIVA para NIM: extrae con regex solo el bloque PENSAMIENTO/ACCION
    y lo reconstruye como PENSAMIENTO/ACCION/FIN, descartando todo lo demás. Si no
    hay PENSAMIENTO, intenta aislar al menos la línea ACCION. Si no reconoce nada,
    devuelve el texto tal cual (que lo intente el parser general).
    """
    if not texto:
        return texto
    texto = texto.replace("*", "")               # quita markdown (negritas/viñetas)
    m = _RE_BLOQUE_FORMATO.search(texto)
    if m:
        pens = " ".join(m.group("pens").split())
        acc = _limpiar_accion(m.group("acc"))
        return f"PENSAMIENTO: {pens}\nACCION: {acc}\nFIN"
    m2 = _RE_SOLO_ACCION.search(texto)
    if m2:
        return f"ACCION: {_limpiar_accion(m2.group('acc'))}\nFIN"
    return texto


class LimiteAPIError(Exception):
    """Se alcanzó el límite de la API de Gemini (HTTP 429 / RESOURCE_EXHAUSTED)."""


@dataclass
class Decision:
    """Respuesta parseada del modelo."""
    pensamiento: str
    accion: str
    raw: str = ""
    modelo: str = MODELO_GEMINI   # qué modelo la produjo (GEMINI / NIM-LLAMA / NIM-MINIMAX)

    @property
    def valida(self) -> bool:
        return bool(self.accion)


# ─── System instruction (formato rígido + comandos válidos) ───────────────────
SYSTEM_INSTRUCTION = f"""\
Eres {AGENT_NAME}, una operadora experta que controla un PC con Windows mirando
capturas de pantalla. Eres autónoma, directa y rápida. Sin charla ni relleno.

OBJETIVO: cumplir la tarea del usuario ejecutando UNA acción por turno. Tras cada
acción verás una NUEVA captura y decidirás la siguiente acción.

FORMATO DE RESPUESTA OBLIGATORIO (sin excepciones):
PENSAMIENTO: <razonamiento breve, máximo 2 líneas>
ACCION: <un solo comando válido>
FIN

COMANDOS VÁLIDOS (uno solo por respuesta):
launch_app NOMBRE       abre una app por su nombre — ej: launch_app notepad
click X Y               clic izquierdo — ej: click 640 360
double_click X Y        doble clic — ej: double_click 120 200
right_click X Y         clic derecho (menú contextual) — ej: right_click 500 300
middle_click X Y        clic central — ej: middle_click 640 360
hover X Y               mueve el cursor sin clicar (menús al pasar) — ej: hover 800 40
drag X1 Y1 X2 Y2        arrastra de un punto a otro — ej: drag 100 100 400 400
type TEXTO              escribe el texto exacto — ej: type Hola mundo
key TECLA               pulsa una tecla — ej: key enter
hotkey A+B              combinación de teclas — ej: hotkey ctrl+s
hold_key MOD+click X Y  mantén MOD (shift/ctrl/alt) y clica — ej: hold_key shift+click 300 400
hold_key MOD+key T      mantén MOD y pulsa una tecla — ej: hold_key ctrl+key a
find_text "TEXTO"       localiza un texto y te devuelve sus coords — ej: find_text "Guardar"
find_image RUTA         localiza un PNG y te devuelve sus coords — ej: find_image C:\\ui\\ok.png
focus_window "TITULO"   trae una ventana al frente — ej: focus_window "Bloc de notas"
scroll up N             desplaza hacia arriba N — ej: scroll up 5
scroll down N           desplaza hacia abajo N — ej: scroll down 5
hscroll DIR N           scroll horizontal (left/right) — ej: hscroll right 5
wait N                  espera N segundos — ej: wait 2
done                    úsalo SOLO cuando la tarea esté completamente terminada

REGLAS CRÍTICAS:
1. UN SOLO comando en la línea ACCION. Jamás dos comandos ni listas (1., 2., -).
2. COORDENADAS: las X Y son píxeles de la IMAGEN que recibes (su tamaño se indica
   en el mensaje). Lee la posición real del elemento en la captura.
3. PARA ABRIR PROGRAMAS usa launch_app (ej: launch_app calc). NO uses win+r ni
   otras hotkeys del sistema: están BLOQUEADAS por seguridad. Reserva el clic para
   botones sin atajo.
4. ¿No estás segura de las coordenadas? Usa find_text "etiqueta" (o find_image
   ruta.png): te devolverán las coords y luego harás click X Y sobre ellas.
5. PENSAMIENTO máximo 2 líneas. Nada de explicaciones largas.
6. Si la tarea ya está hecha, responde con ACCION: done.
7. Termina SIEMPRE con FIN en su propia línea.
"""

# System prompt para los modelos de fallback NIM (propensos a divagar, usar markdown
# o copiar la plantilla de comandos). Anteponemos una orden tajante de formato.
NIM_SYSTEM_INSTRUCTION = (
    "RESPONDE SOLO con el bloque PENSAMIENTO/ACCION/FIN. Máximo 3 líneas en total. "
    "Nada más. TEXTO PLANO: prohibido markdown, asteriscos, negritas, viñetas o listas. "
    "En ACCION pon UN comando con números REALES leídos de la imagen "
    "(nunca escribas literalmente 'X Y').\n\n"
) + SYSTEM_INSTRUCTION

# Prompt one-shot para confirmar 'done' (FIX #3): rol de verificador, no de operadora.
_CONFIRM_SYS = ("Eres un verificador estricto. Mira la captura y responde SOLO con "
                "'SI' o 'NO' seguido de 3-5 palabras de motivo. Nada de acciones.")

# ─── Parsers de la respuesta ──────────────────────────────────────────────────
_RE_PENSAMIENTO = re.compile(r"PENSAMIENTO\s*:\s*(.+?)(?:\n\s*ACCI[OÓ]N\s*:|\Z)",
                             re.I | re.S)
_RE_ACCION      = re.compile(r"ACCI[OÓ]N\s*:\s*(.+?)(?:\n|FIN|\Z)", re.I | re.S)
# Respaldo: si el modelo omite el prefijo, reconoce un comando suelto.
_RE_COMANDO_SUELTO = re.compile(
    r"^(?:launch_app|double_click|right_click|middle_click|hold_key|find_text|"
    r"find_image|focus_window|hscroll|click|type|key|hotkey|scroll|drag|hover|"
    r"wait|done)\b.*$", re.I | re.M
)

# Verbos de comando para detectar acciones encadenadas ("type calc key enter").
# Lista EXACTA (a propósito NO incluye 'scroll'). Si tras el primer token aparece
# otro de estos verbos, se conserva solo el primer comando y se descarta el resto.
_VERBOS_ACCION = ("click", "double_click", "type", "key", "hotkey", "wait", "done")
# Tras 'type' el argumento es TEXTO LIBRE que puede contener 'click', 'done', etc.
# de forma legítima ("type Haz click en guardar"): solo cortan key/hotkey, que es
# lo que el modelo encadena en la práctica ('type calc key enter').
_VERBOS_TRAS_TYPE = ("key", "hotkey")


def _un_solo_comando(accion: str) -> str:
    """Conserva SOLO el primer comando si la ACCION encadena varios.

    El modelo a veces junta comandos en una línea ('type calc key enter',
    'hotkey win+r type calc key enter'). Se recorta en el primer verbo de
    `_VERBOS_ACCION` que aparezca DESPUÉS del primer token y se descarta el resto.
    """
    if not accion:
        return accion
    tokens = accion.split()
    verbos = _VERBOS_TRAS_TYPE if tokens[0].lower() == "type" else _VERBOS_ACCION
    for i in range(1, len(tokens)):
        if tokens[i].lower() in verbos:
            recortada = " ".join(tokens[:i])
            descartado = " ".join(tokens[i:])
            logger.warning("Parser: varios comandos en una acción — conservo '%s', "
                           "descarto '%s'.", recortada, descartado)
            return recortada
    return accion


def parsear(texto: str) -> Decision:
    """Extrae PENSAMIENTO y ACCION de la respuesta cruda, de forma tolerante."""
    raw = (texto or "").strip()
    if not raw:
        return Decision(pensamiento="", accion="", raw=raw)

    # Los backticks/cercas de código nunca forman parte de un comando: se quitan
    # antes de aplicar los patrones (así el respaldo sin-prefijo ancla bien).
    limpio = raw.replace("`", "")

    mp = _RE_PENSAMIENTO.search(limpio)
    pensamiento = (mp.group(1).strip() if mp else "").replace("\n", " ").strip()

    ma = _RE_ACCION.search(limpio)
    accion = ma.group(1).strip() if ma else ""

    if not accion:
        # Respaldo: el modelo escribió el comando sin el prefijo ACCION.
        ms = _RE_COMANDO_SUELTO.search(limpio)
        accion = ms.group(0).strip() if ms else ""

    accion = accion.strip().rstrip(".")
    accion = _un_solo_comando(accion)              # FIX-2: solo el primer comando
    return Decision(pensamiento=pensamiento, accion=accion, raw=raw)


class Cerebro:
    """
    Interfaz con Gemini. Mantiene el historial mínimo de la tarea en curso en el
    formato nativo de la API (contents) y aplica GC de imágenes antes de enviar.
    """

    def __init__(self) -> None:
        self._historial: list[dict] = []
        self._timeout = httpx.Timeout(
            connect=TIMEOUT_CONNECT, read=TIMEOUT_READ,
            write=TIMEOUT_WRITE, pool=TIMEOUT_POOL,
        )
        self._cliente = httpx.Client(timeout=self._timeout)
        # Modelo que respondió la última llamada (GEMINI / NIM-LLAMA / NIM-MINIMAX).
        self.ultimo_modelo = MODELO_GEMINI
        # Modelo NIM concreto usado en el último fallback (para logs/diagnóstico).
        self.ultimo_nim = ""
        # Memoria persistente: inyecta las lecciones de sesiones anteriores en el
        # system prompt una sola vez al iniciar (Gemini no aprende entre sesiones).
        seccion = lecciones.seccion_prompt()
        self._sys = SYSTEM_INSTRUCTION + seccion
        self._nim_sys = NIM_SYSTEM_INSTRUCTION + seccion
        if seccion:
            logger.info("Lecciones cargadas en el prompt (%d).", len(lecciones.cargar()))
        # El rate limiter ahora es CROSS-PROCESS (módulo, archivo compartido).
        logger.info("Cerebro iniciado — modelo: %s.", GEMINI_MODEL)

    # ── API pública ──────────────────────────────────────────────────────────
    def pensar(
        self,
        objetivo_turno: str,
        imagen_b64: Optional[str] = None,
        profundo: bool = False,
    ) -> Decision:
        """
        Añade un turno (texto + imagen) al historial, llama a Gemini y devuelve la
        Decision parseada. `profundo=True` activa el presupuesto de pensamiento
        (estado THINKING); en WORKING/OVERLOADED debe ser False para ir al máximo.

        Lanza `LimiteAPIError` si la API responde 429.
        """
        self._historial.append(self._mensaje_usuario(objetivo_turno, imagen_b64))
        self._podar()

        raw = self._llamar(profundo)
        if raw:
            self._historial.append({"role": "model", "parts": [{"text": raw}]})

        decision = parsear(raw)
        decision.modelo = self.ultimo_modelo      # marca qué modelo la produjo
        return decision

    def registrar_resultado(self, nota: str) -> None:
        """Inyecta una nota de Sistema (resultado de la acción) como turno de usuario."""
        if nota:
            self._historial.append({"role": "user", "parts": [{"text": nota}]})
            self._podar()

    def confirmar_done(self, objetivo: str, imagen_b64: Optional[str]) -> bool:
        """FIX #3: confirma un 'done' antes de aceptarlo. Pregunta a Gemini, sobre una
        captura FRESCA, si la tarea se ve completada (rol verificador). Devuelve False
        SOLO ante un 'NO' claro; ante SI / respuesta ambigua acepta (lenient: el
        verificador del trainer es el juez final). Propaga LimiteAPIError."""
        self._historial.append(self._mensaje_usuario(
            f"VERIFICACIÓN (no emitas acciones): ¿la tarea «{objetivo}» está COMPLETADA "
            "y visible en esta captura? Responde SOLO 'SI' o 'NO' + motivo breve.",
            imagen_b64))
        self._podar()
        sys_g, nim_g = self._sys, self._nim_sys
        self._sys = self._nim_sys = _CONFIRM_SYS          # rol verificador durante la llamada
        try:
            raw = self._llamar(profundo=False)
        finally:
            self._sys, self._nim_sys = sys_g, nim_g       # restaurar SIEMPRE
        if raw:
            self._historial.append({"role": "model", "parts": [{"text": raw}]})
        m = re.search(r"\b(s[ií]|no)\b", raw or "", re.I)
        confirmado = True if not m else (m.group(1).lower() != "no")
        logger.info("Confirmación 'done': %s (%s)", "SI" if confirmado else "NO",
                    (raw or "")[:50].replace("\n", " "))
        return confirmado

    def reset(self) -> None:
        """Vacía el historial de la tarea en curso."""
        self._historial.clear()

    def cerrar(self) -> None:
        try:
            self._cliente.close()
        except Exception:                          # noqa: BLE001
            pass

    # ── Persistencia (para parada limpia / reanudación) ────────────────────────
    def exportar_historial(self) -> list[dict]:
        """Historial SIN imágenes (las capturas son volátiles; se re-perciben)."""
        return [self._sin_imagen(m) for m in self._historial]

    def importar_historial(self, historial: list[dict]) -> None:
        self._historial = [dict(m) for m in (historial or [])]
        logger.info("Historial restaurado (%d turnos).", len(self._historial))

    # ── Construcción de mensajes ───────────────────────────────────────────────
    @staticmethod
    def _mensaje_usuario(texto: str, imagen_b64: Optional[str]) -> dict:
        partes: list[dict] = [{"text": texto}]
        if imagen_b64:
            partes.append({
                "inlineData": {"mimeType": "image/jpeg", "data": imagen_b64}
            })
        return {"role": "user", "parts": partes}

    @staticmethod
    def _sin_imagen(msg: dict) -> dict:
        partes = [p for p in msg.get("parts", []) if "inlineData" not in p]
        return {"role": msg.get("role", "user"), "parts": partes or [{"text": ""}]}

    def _podar(self) -> None:
        """Mantiene el historial MÍNIMO (últimos MAX_HISTORIAL turnos)."""
        if len(self._historial) > MAX_HISTORIAL:
            self._historial = self._historial[-MAX_HISTORIAL:]

    def _gc_imagenes(self) -> list[dict]:
        """
        Recolector de basura visual: deja la imagen SOLO en el último turno de
        usuario. Las capturas viejas no aportan y disparan tokens/latencia.
        """
        if not self._historial:
            return []
        # Índice del último mensaje de usuario (el de la captura fresca).
        ultimo_user = max(
            (i for i, m in enumerate(self._historial) if m.get("role") == "user"),
            default=-1,
        )
        salida: list[dict] = []
        for i, m in enumerate(self._historial):
            salida.append(m if i == ultimo_user else self._sin_imagen(m))
        return salida

    # ── Orquestación de la llamada (Gemini + fallback opcional) ─────────────────
    def _llamar(self, profundo: bool) -> str:
        """
        Llama al modelo principal (Gemini). Devuelve el texto, "" si hubo error
        recuperable, o lanza `LimiteAPIError` si se alcanzó el límite (429).

        FALLBACK TEMPORAL DE ENTRENAMIENTO (fácil de retirar): si Gemini da 429 y
        TRAINING_MODE está activo con NVIDIA_API_KEY, prueba NVIDIA NIM antes de
        rendirse. En producción (TRAINING_MODE=false) el 429 se propaga tal cual,
        y el orquestador guarda estado y se detiene.

        Para quitar el fallback más adelante: borra el bloque `if TRAINING_MODE…`
        y el método `_llamar_nim`. Nada más depende de ellos.
        """
        try:
            texto = self._llamar_gemini(profundo)
            self.ultimo_modelo = MODELO_GEMINI
            return texto
        except LimiteAPIError:
            if TRAINING_MODE and NVIDIA_API_KEY:
                # Cadena NIM: se prueba cada modelo en orden hasta que uno responda.
                for modelo in NVIDIA_FALLBACK_MODELS:
                    logger.warning("Gemini 429 → fallback NIM '%s' [TRAINING_MODE].", modelo)
                    texto = self._llamar_nim(modelo, profundo)
                    if texto:
                        self.ultimo_modelo = _etiqueta_nim(modelo)
                        self.ultimo_nim = modelo
                        return texto
                    logger.warning("NIM '%s' sin respuesta — probando el siguiente.", modelo)
                logger.error("Todos los modelos NIM fallaron — se detiene.")
            # Producción, fallback desactivado o cadena NIM agotada → propagar.
            raise

    # ── Rate limiter de Gemini (15 RPM → ≥ 4.5 s entre llamadas) ────────────────
    def _rate_limit_gemini(self) -> None:
        """Coordina ≥ GEMINI_MIN_INTERVALO s entre llamadas a Gemini, COMPARTIDO con
        el entrenador vía tasks/rate_gemini.json (mutex de exclusión mutua)."""
        _rate_limit_compartido()

    # ── Llamada HTTP a Gemini (principal) ───────────────────────────────────────
    def _llamar_gemini(self, profundo: bool) -> str:
        """POST a generateContent. "" si error recuperable; lanza LimiteAPIError en 429."""
        self._rate_limit_gemini()
        budget = THINK_BUDGET_PROFUNDO if profundo else THINK_BUDGET_RAPIDO
        # Con pensamiento activo, la respuesta visible necesita tokens aparte.
        max_tokens = GEN_MAX_TOKENS + (THINK_RESPUESTA_EXTRA + budget if budget else 0)

        payload = {
            "systemInstruction": {"parts": [{"text": self._sys}]},
            "contents": self._gc_imagenes(),
            "generationConfig": {
                "temperature": GEN_TEMPERATURE,
                "topP": GEN_TOP_P,
                "maxOutputTokens": max_tokens,
                "stopSequences": GEN_STOP,
                "thinkingConfig": {"thinkingBudget": budget},
            },
        }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY,
        }

        try:
            resp = self._cliente.post(GEMINI_URL, json=payload, headers=headers)
        except httpx.TimeoutException:
            logger.error("Gemini: timeout de red.")
            return ""
        except httpx.HTTPError as exc:
            logger.error("Gemini: error de red — %s", exc)
            return ""

        if resp.status_code == 429:
            logger.warning("Gemini: 429 — límite de API alcanzado.")
            raise LimiteAPIError("HTTP 429 RESOURCE_EXHAUSTED")

        if resp.status_code != 200:
            cuerpo = resp.text[:300]
            # Algunos despliegues devuelven RESOURCE_EXHAUSTED con otro código.
            if "RESOURCE_EXHAUSTED" in cuerpo:
                raise LimiteAPIError(f"HTTP {resp.status_code} RESOURCE_EXHAUSTED")
            logger.error("Gemini: HTTP %d — %s", resp.status_code, cuerpo)
            return ""

        return self._extraer_texto(resp.json())

    # ── Fallback NVIDIA NIM (TEMPORAL, API estilo OpenAI) ───────────────────────
    def _llamar_nim(self, modelo: str, profundo: bool) -> str:
        """
        Llama a un modelo de NVIDIA NIM (chat/completions estilo OpenAI) con un
        system prompt de formato estricto + el historial. Devuelve el texto YA
        AISLADO al bloque PENSAMIENTO/ACCION/FIN (limpieza agresiva), o "" si falla.
        Sirve tanto para el multimodal (Llama) como para el segundo modelo (MiniMax M3).
        """
        mensajes = self._a_openai(self._gc_imagenes(), self._nim_sys)
        payload = {
            "model": modelo,
            "messages": mensajes,
            "temperature": GEN_TEMPERATURE,
            "top_p": GEN_TOP_P,
            "max_tokens": _NIM_MAX_TOKENS,
            "stop": GEN_STOP,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            resp = self._cliente.post(NVIDIA_API_URL, json=payload,
                                      headers=headers, timeout=_NIM_TIMEOUT)
        except httpx.HTTPError as exc:
            logger.error("NIM '%s': error de red — %s", modelo, exc)
            return ""

        if resp.status_code != 200:
            logger.error("NIM '%s': HTTP %d — %s", modelo, resp.status_code, resp.text[:200])
            return ""

        # Limpieza AGRESIVA: aísla el bloque de formato y descarta el ruido JSON.
        return _aislar_formato(self._extraer_openai(resp.json()))

    @staticmethod
    def _a_openai(contents: list[dict], system: str = SYSTEM_INSTRUCTION) -> list[dict]:
        """Convierte el historial (formato Gemini) a `messages` de OpenAI/NIM."""
        mensajes: list[dict] = [{"role": "system", "content": system}]
        for m in contents:
            rol = "assistant" if m.get("role") == "model" else "user"
            piezas: list[dict] = []
            for p in m.get("parts", []):
                if "text" in p:
                    piezas.append({"type": "text", "text": p["text"]})
                elif "inlineData" in p:
                    d = p["inlineData"]
                    url = f"data:{d.get('mimeType', 'image/jpeg')};base64,{d.get('data', '')}"
                    piezas.append({"type": "image_url", "image_url": {"url": url}})
            if not piezas:
                continue
            # Si solo hay texto, se colapsa a string (más compatible con NIM).
            if len(piezas) == 1 and piezas[0]["type"] == "text":
                mensajes.append({"role": rol, "content": piezas[0]["text"]})
            else:
                mensajes.append({"role": rol, "content": piezas})
        return mensajes

    @staticmethod
    def _extraer_openai(datos: dict) -> str:
        """
        Extrae el texto de una respuesta chat/completions, limpiando <think>.

        Salvaguarda: este modelo razonador a veces deja `content` vacío (agota el
        presupuesto razonando, finish=length) pero escribe el bloque de formato en
        `reasoning_content`. Si ahí hay un marcador ACCION explícito, lo rescatamos.
        """
        try:
            ch = datos["choices"][0]
            msg = ch.get("message", {})
            texto = _limpiar_think(msg.get("content") or "").strip()
            if texto:
                return texto
            razon = _limpiar_think(msg.get("reasoning_content") or "").strip()
            if re.search(r"ACCI[OÓ]N\s*[\"']?\s*:", razon, re.I):
                logger.warning("NIM: contenido vacío (finish=%s) — rescatando ACCION "
                               "del razonamiento.", ch.get("finish_reason"))
                return razon
            logger.warning("NIM: contenido vacío y sin ACCION en el razonamiento "
                           "(finish=%s).", ch.get("finish_reason"))
            return ""
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("NIM: respuesta no parseable — %s", exc)
            return ""

    @staticmethod
    def _extraer_texto(datos: dict) -> str:
        """Concatena el texto de las partes del primer candidato. "" si no hay."""
        try:
            candidatos = datos.get("candidates", [])
            if not candidatos:
                # promptFeedback con blockReason → contenido bloqueado.
                fb = datos.get("promptFeedback", {})
                if fb:
                    logger.warning("Gemini: sin candidatos (feedback: %s).", fb)
                return ""
            cand = candidatos[0]
            finish = cand.get("finishReason", "")
            partes = cand.get("content", {}).get("parts", [])
            texto = "".join(p.get("text", "") for p in partes).strip()
            if not texto and finish == "MAX_TOKENS":
                logger.warning("Gemini: MAX_TOKENS sin texto visible "
                               "(el pensamiento consumió el presupuesto).")
            return texto
        except Exception as exc:                   # noqa: BLE001
            logger.error("Gemini: respuesta no parseable — %s", exc)
            return ""
