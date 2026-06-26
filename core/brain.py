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
    NVIDIA_MODEL,
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

# Tokens máximos para el fallback NIM (modelo de razonamiento: necesita margen
# para su cadena de pensamiento antes de la respuesta en formato).
_NIM_MAX_TOKENS = 1024
# Limpia los bloques de razonamiento <think>…</think> que emiten los modelos de
# razonamiento, dejando solo la respuesta en formato PENSAMIENTO/ACCION.
_RE_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_RE_THINK_ABIERTO = re.compile(r"<think>.*\Z", re.S | re.I)


def _limpiar_think(texto: str) -> str:
    """Quita bloques <think> (cerrados o sin cerrar) de la salida de un modelo razonador."""
    if "<think>" not in texto.lower():
        return texto
    texto = _RE_THINK.sub("", texto)
    texto = _RE_THINK_ABIERTO.sub("", texto)   # <think> sin cierre (cortado) → fuera
    return texto.strip()


class LimiteAPIError(Exception):
    """Se alcanzó el límite de la API de Gemini (HTTP 429 / RESOURCE_EXHAUSTED)."""


@dataclass
class Decision:
    """Respuesta parseada del modelo."""
    pensamiento: str
    accion: str
    raw: str = ""

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
click X Y           clic izquierdo en el punto (X, Y) de la imagen
double_click X Y    doble clic en (X, Y)
type TEXTO          escribe el texto exacto indicado
key TECLA           pulsa una tecla (key enter / key esc / key tab / key f5)
hotkey A+B          combinación (hotkey win+r / hotkey ctrl+c / hotkey alt+f4)
wait N              espera N segundos (entero)
done                úsalo SOLO cuando la tarea esté completamente terminada

REGLAS CRÍTICAS:
1. UN SOLO comando en la línea ACCION. Jamás dos comandos ni listas (1., 2., -).
2. COORDENADAS: las X Y son píxeles de la IMAGEN que recibes (su tamaño se indica
   en el mensaje). Lee la posición real del elemento en la captura.
3. PRIORIDAD AL TECLADO: para abrir programas o buscar usa primero el teclado
   (hotkey win+r → type programa → key enter). Reserva el clic para botones sin atajo.
4. PENSAMIENTO máximo 2 líneas. Nada de explicaciones largas.
5. Si la tarea ya está hecha, responde con ACCION: done.
6. Termina SIEMPRE con FIN en su propia línea.
"""

# ─── Parsers de la respuesta ──────────────────────────────────────────────────
_RE_PENSAMIENTO = re.compile(r"PENSAMIENTO\s*:\s*(.+?)(?:\n\s*ACCI[OÓ]N\s*:|\Z)",
                             re.I | re.S)
_RE_ACCION      = re.compile(r"ACCI[OÓ]N\s*:\s*(.+?)(?:\n|FIN|\Z)", re.I | re.S)
# Respaldo: si el modelo omite el prefijo, reconoce un comando suelto.
_RE_COMANDO_SUELTO = re.compile(
    r"^(?:click|double_click|type|key|hotkey|wait|done)\b.*$", re.I | re.M
)


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

        return parsear(raw)

    def registrar_resultado(self, nota: str) -> None:
        """Inyecta una nota de Sistema (resultado de la acción) como turno de usuario."""
        if nota:
            self._historial.append({"role": "user", "parts": [{"text": nota}]})
            self._podar()

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
            return self._llamar_gemini(profundo)
        except LimiteAPIError:
            if TRAINING_MODE and NVIDIA_API_KEY:
                logger.warning("Gemini 429 → fallback a NVIDIA NIM (%s) [TRAINING_MODE].",
                               NVIDIA_MODEL)
                texto = self._llamar_nim(profundo)
                if texto:
                    return texto
                logger.error("NIM también falló — sin fallback restante; se detiene.")
            # Producción, fallback desactivado o NIM agotado → propagar (guardar y parar).
            raise

    # ── Llamada HTTP a Gemini (principal) ───────────────────────────────────────
    def _llamar_gemini(self, profundo: bool) -> str:
        """POST a generateContent. "" si error recuperable; lanza LimiteAPIError en 429."""
        budget = THINK_BUDGET_PROFUNDO if profundo else THINK_BUDGET_RAPIDO
        # Con pensamiento activo, la respuesta visible necesita tokens aparte.
        max_tokens = GEN_MAX_TOKENS + (THINK_RESPUESTA_EXTRA + budget if budget else 0)

        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
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
    def _llamar_nim(self, profundo: bool) -> str:
        """
        Llama a NVIDIA NIM (formato chat/completions de OpenAI) con el MISMO system
        prompt e historial (convertidos al formato OpenAI). Devuelve el texto en el
        formato PENSAMIENTO/ACCION, o "" si falla (incluido 429 de NIM). Nunca lanza.
        """
        mensajes = self._a_openai(self._gc_imagenes())
        payload = {
            "model": NVIDIA_MODEL,
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
            resp = self._cliente.post(NVIDIA_API_URL, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            logger.error("NIM: error de red — %s", exc)
            return ""

        if resp.status_code != 200:
            logger.error("NIM: HTTP %d — %s", resp.status_code, resp.text[:200])
            return ""

        return self._extraer_openai(resp.json())

    @staticmethod
    def _a_openai(contents: list[dict]) -> list[dict]:
        """Convierte el historial (formato Gemini) a `messages` de OpenAI/NIM."""
        mensajes: list[dict] = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
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
        """Extrae el texto de una respuesta chat/completions, limpiando <think>."""
        try:
            msg = datos["choices"][0]["message"]
            contenido = msg.get("content") or ""
            return _limpiar_think(contenido).strip()
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
