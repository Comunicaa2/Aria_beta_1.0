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

import logging
import re
from dataclasses import dataclass
from typing import Optional

import httpx

from compartido import (MODELO_GEMINI, MODELO_NIM_LLAMA, MODELO_NIM_OMNI,
                        rate_limit_compartido)
from core import lecciones, memoria, skills
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

# Los identificadores de modelo (MODELO_GEMINI/NIM_*) y el rate limiter
# cross-process viven en compartido.py (una sola fuente de verdad con el
# entrenador).


def _etiqueta_nim(modelo: str) -> str:
    """Mapea el id de un modelo NIM a su etiqueta corta (robusto a overrides)."""
    return MODELO_NIM_LLAMA if "llama" in modelo.lower() else MODELO_NIM_OMNI

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
# Bloque CONTENIDO (multilínea) de la acción 'guardar': todo lo que sigue a la
# etiqueta hasta FIN o el final (GEN_STOP corta en FIN, así que suele ser el final).
_RE_CONTENIDO = re.compile(r"CONTENIDO\s*:\s*\n?(?P<cont>.*?)(?:\n\s*FIN\s*$|\Z)",
                           re.I | re.S)


def _extraer_contenido(raw: str) -> str:
    """Extrae el bloque CONTENIDO del texto CRUDO (sin limpiar: el contenido de un
    archivo puede llevar backticks, llaves, etc.). Quita cercas de código ``` si
    el modelo envolvió el contenido en ellas."""
    m = _RE_CONTENIDO.search(raw or "")
    if not m:
        return ""
    cont = m.group("cont").strip("\r\n")
    lineas = cont.split("\n")
    if lineas and lineas[0].strip().startswith("```"):
        lineas = lineas[1:]
    if lineas and lineas[-1].strip() == "```":
        lineas = lineas[:-1]
    return "\n".join(lineas)


def _limpiar_think(texto: str) -> str:
    """Quita bloques <think> (cerrados o sin cerrar) de la salida de un modelo razonador."""
    if "<think>" not in texto.lower():
        return texto
    texto = _RE_THINK.sub("", texto)
    texto = _RE_THINK_ABIERTO.sub("", texto)   # <think> sin cierre (cortado) → fuera
    return texto.strip()


def _limpiar_accion(acc: str) -> str:
    """Recorta basura JSON/comillas del valor de ACCION (p. ej. "done'}]" → "done").
    Solo quita una comilla de extremo si NO tiene pareja (conteo impar): así
    'click_ui "Cerrar"' conserva sus comillas intactas."""
    acc = re.split(r"[\r\n}\]]", acc)[0].strip()     # corta en el 1.er ruido estructural
    if len(acc) >= 2 and acc[0] == acc[-1] and acc[0] in "'\"":
        acc = acc[1:-1]                              # acción ENVUELTA en comillas
    if acc and acc[0] in "'\"" and acc.count(acc[0]) % 2 == 1:
        acc = acc[1:]
    if acc and acc[-1] in "'\"" and acc.count(acc[-1]) % 2 == 1:
        acc = acc[:-1]
    return acc.rstrip("}],) ").strip()


def _aislar_formato(texto: str) -> str:
    """
    Limpieza AGRESIVA para NIM: extrae con regex solo el bloque PENSAMIENTO/ACCION
    y lo reconstruye como PENSAMIENTO/ACCION/FIN, descartando todo lo demás. Si no
    hay PENSAMIENTO, intenta aislar al menos la línea ACCION. Si no reconoce nada,
    devuelve el texto tal cual (que lo intente el parser general).
    """
    if not texto:
        return texto
    # El CONTENIDO se extrae del texto ORIGINAL: la limpieza de '*' corrompería
    # código (Python/markdown usan asteriscos legítimamente).
    cont = _extraer_contenido(texto)
    bloque_cont = f"\nCONTENIDO:\n{cont}" if cont else ""
    texto = texto.replace("*", "")               # quita markdown (negritas/viñetas)
    m = _RE_BLOQUE_FORMATO.search(texto)
    if m:
        pens = " ".join(m.group("pens").split())
        acc = _limpiar_accion(m.group("acc"))
        return f"PENSAMIENTO: {pens}\nACCION: {acc}{bloque_cont}\nFIN"
    m2 = _RE_SOLO_ACCION.search(texto)
    if m2:
        return f"ACCION: {_limpiar_accion(m2.group('acc'))}{bloque_cont}\nFIN"
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
    contenido: str = ""           # bloque CONTENIDO (multilínea) para la acción 'guardar'

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

Excepción ÚNICA: la acción 'guardar' añade un bloque CONTENIDO antes de FIN:
PENSAMIENTO: <breve>
ACCION: guardar nombre.ext
CONTENIDO:
<contenido COMPLETO del archivo — aquí sí puedes usar varias líneas>
FIN

COMANDOS VÁLIDOS (uno solo por respuesta):
launch_app NOMBRE       abre una app por su nombre — ej: launch_app notepad
click_ui "NOMBRE"       clica un elemento de la ventana ACTIVA por su nombre visible
                        (botón, menú, campo) — ej: click_ui "Guardar"
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
guardar NOMBRE          crea/sobrescribe un archivo en la carpeta de trabajo con el
                        bloque CONTENIDO — ej: guardar analisis.py / guardar informe.md
ejecutar_python NOMBRE [ARGS]  ejecuta un .py de la carpeta de trabajo (con argumentos
                        opcionales) y te devuelve su salida (print) en el siguiente
                        turno — ej: ejecutar_python analisis.py datos.csv
done                    úsalo SOLO cuando la tarea esté completamente terminada

REGLAS CRÍTICAS:
1. UN SOLO comando en la línea ACCION. Jamás dos comandos ni listas (1., 2., -).
2. COORDENADAS: las X Y son píxeles de la IMAGEN que recibes (su tamaño se indica
   en el mensaje). Lee la posición real del elemento en la captura.
3. PARA ABRIR PROGRAMAS usa launch_app (ej: launch_app calc). NO uses win+r ni
   otras hotkeys del sistema: están BLOQUEADAS por seguridad. Reserva el clic para
   botones sin atajo.
4. PARA CLICAR ELEMENTOS CON TEXTO/NOMBRE (botones, menús, opciones) PREFIERE
   click_ui "nombre" con el nombre EXACTO visible del elemento. Puede fallar si
   el nombre no coincide, es ambiguo o la app no expone su UI al sistema. Si
   click_ui falla, o si repites el mismo PENSAMIENTO 2 turnos seguidos sin
   avanzar, CAMBIA de estrategia: usa find_text "etiqueta" (o find_image
   ruta.png) y luego click X Y sobre las coords devueltas. No insistas con lo
   que ya falló.
5. PENSAMIENTO máximo 2 líneas. Nada de explicaciones largas.
6. Si la tarea ya está hecha, responde con ACCION: done.
7. Termina SIEMPRE con FIN en su propia línea.
8. PARA ANALIZAR DATOS O PROGRAMAR (métricas, estadística, gráficos, cálculos):
   NO hagas cuentas de cabeza ni a ojo. Flujo: guardar script.py (con el código en
   CONTENIDO) → ejecutar_python script.py → lee la salida real → guardar informe.md
   con las conclusiones. En 'guardar' usa SOLO nombres de archivo simples (sin rutas
   ni barras). Si el script falla verás el error: corrígelo y vuelve a guardarlo.
9. SKILLS (automejora): si en SKILLS DISPONIBLES hay una que resuelva (parte de)
   la tarea, úsala en vez de reescribir el código. Si resuelves algo que volverá a
   pedirse, consérvalo como skill: guardar skill_nombre.py cuyo docstring (una
   línea) diga qué hace y qué argumentos recibe. Si una skill falla o queda lenta,
   guárdala corregida/mejorada con el MISMO nombre: así te optimizas a ti misma.
10. SUPERA LO PEDIDO: cumple el objetivo de la mejor forma posible (más completo,
   más ordenado, más útil), sin salirte de él ni tocar nada no relacionado. Si
   tras cumplirlo el Sistema te ofrece una fase EXTRA, propone solo mejoras
   pequeñas y seguras relacionadas con la tarea, o 'done' si no las hay.
"""

# System prompt para los modelos de fallback NIM (propensos a divagar, usar markdown
# o copiar la plantilla de comandos). Anteponemos una orden tajante de formato.
NIM_SYSTEM_INSTRUCTION = (
    "RESPONDE SOLO con el bloque PENSAMIENTO/ACCION/FIN. Máximo 3 líneas en total "
    "(única excepción: el bloque CONTENIDO de la acción 'guardar'). "
    "Nada más. TEXTO PLANO: prohibido markdown, asteriscos, negritas, viñetas o listas. "
    "En ACCION pon UN comando con números REALES leídos de la imagen "
    "(nunca escribas literalmente 'X Y').\n\n"
) + SYSTEM_INSTRUCTION

# Prompt one-shot para confirmar 'done' (FIX #3): rol de verificador, no de operadora.
# Las acciones de archivo (guardar/ejecutar_python) NO se ven en pantalla: el
# verificador debe juzgarlas por las notas del Sistema en el historial, no exigir
# evidencia visual imposible (bug del E2E 2026-07-19: rechazaba tareas ya hechas).
_CONFIRM_SYS = ("Eres un verificador estricto. Decide con la captura Y el historial: "
                "las acciones de archivos (guardar, ejecutar_python) no se ven en "
                "pantalla — júzgalas por las notas del Sistema en la conversación. "
                "Responde SOLO con 'SI' o 'NO' seguido de 3-5 palabras de motivo. "
                "Nada de acciones.")

# Prompt one-shot para destilar una lección (tarea fallida o completada lenta).
_LECCION_SYS = ("Eres la memoria de Aria. Responde SOLO con UNA regla breve y "
                "GENERAL (máximo 15 palabras) que mejore el desempeño futuro según "
                "lo descrito. Sin prefijos, sin formato, sin acciones.")

# ─── Parsers de la respuesta ──────────────────────────────────────────────────
_RE_PENSAMIENTO = re.compile(r"PENSAMIENTO\s*:\s*(.+?)(?:\n\s*ACCI[OÓ]N\s*:|\Z)",
                             re.I | re.S)
_RE_ACCION      = re.compile(r"ACCI[OÓ]N\s*:\s*(.+?)(?:\n|FIN|\Z)", re.I | re.S)
# Respaldo: si el modelo omite el prefijo, reconoce un comando suelto.
_RE_COMANDO_SUELTO = re.compile(
    r"^(?:launch_app|double_click|right_click|middle_click|hold_key|click_ui|"
    r"find_text|find_image|focus_window|hscroll|click|type|key|hotkey|scroll|"
    r"drag|hover|wait|guardar|ejecutar_python|done)\b.*$", re.I | re.M
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
    # El CONTENIDO se extrae del texto CRUDO (backticks y llaves son legítimos ahí).
    contenido = _extraer_contenido(raw) if accion.lower().startswith("guardar") else ""
    return Decision(pensamiento=pensamiento, accion=accion, raw=raw,
                    contenido=contenido)


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
        # Memoria persistente: lecciones + catálogo de skills en el system prompt.
        self._recargar_memoria()
        logger.info("Cerebro iniciado — modelo: %s.", GEMINI_MODEL)

    def _recargar_memoria(self, objetivo: str = "") -> None:
        """Reconstruye el system prompt con lecciones y skills FRESCAS. Se llama
        al iniciar y en cada reset(): una skill creada en la tarea anterior queda
        disponible en la siguiente sin reiniciar Aria. Con `objetivo`, las
        lecciones entran por relevancia y se añade la EXPERIENCIA RELEVANTE de la
        memoria episódica (RAG) — recuerdos de tareas parecidas, no todo a granel."""
        experiencia = memoria.seccion_prompt(objetivo)
        if experiencia:
            logger.info("Memoria RAG: EXPERIENCIA RELEVANTE inyectada (%d episodios).",
                        experiencia.count("- «"))
        seccion = (lecciones.seccion_prompt(objetivo) + skills.seccion_prompt()
                   + experiencia)
        self._sys = SYSTEM_INSTRUCTION + seccion
        self._nim_sys = NIM_SYSTEM_INSTRUCTION + seccion
        n_skills = len(skills.listar())
        if n_skills:
            logger.info("Memoria en prompt: %d lecciones, %d skills.",
                        len(lecciones.cargar()), n_skills)

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
            f"VERIFICACIÓN (no emitas acciones): ¿la tarea «{objetivo}» está COMPLETADA? "
            "Usa la captura y, para acciones de archivo (guardar/ejecutar_python), las "
            "notas del Sistema del historial. Responde SOLO 'SI' o 'NO' + motivo breve.",
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

    def aprender_leccion(self, objetivo: str, motivo: str) -> str:
        """Destila UNA regla breve de una tarea que terminó sin completarse y la
        persiste en tasks/lecciones.json (se inyectará al prompt en la próxima
        sesión). Devuelve la regla, o '' si no se pudo generar. Nunca lanza:
        aprender es best-effort, no puede tumbar el cierre de la tarea."""
        self._historial.append({"role": "user", "parts": [{"text":
            f"La tarea «{objetivo}» terminó así: {motivo}. Revisa esta conversación "
            "y escribe UNA regla breve y general (máx. 15 palabras) para hacerlo "
            "mejor la próxima vez. Solo la regla."}]})
        self._podar()
        sys_g, nim_g = self._sys, self._nim_sys
        self._sys = self._nim_sys = _LECCION_SYS
        try:
            raw = self._llamar(profundo=False)
        except LimiteAPIError:
            return ""                     # sin cuota no hay lección: no pasa nada
        except Exception:                 # noqa: BLE001
            logger.warning("aprender_leccion: fallo inesperado.", exc_info=True)
            return ""
        finally:
            self._sys, self._nim_sys = sys_g, nim_g
        regla = " ".join(_limpiar_think(raw or "").split()).strip(" .\"'")
        if not regla or re.match(r"(?i)(PENSAMIENTO|ACCI[OÓ]N)\b", regla):
            return ""
        lecciones.registrar(regla)
        return regla

    def reset(self, objetivo: str = "") -> None:
        """Vacía el historial de la tarea en curso y refresca lecciones/skills.
        Con `objetivo`, además recupera la experiencia relevante (memoria RAG)."""
        self._historial.clear()
        self._recargar_memoria(objetivo)

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

    # ── Llamada HTTP a Gemini (principal) ───────────────────────────────────────
    def _llamar_gemini(self, profundo: bool) -> str:
        """POST a generateContent. "" si error recuperable; lanza LimiteAPIError en 429."""
        rate_limit_compartido()                    # compartido con el entrenador
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
