"""core/memoria.py — Memoria episódica RAG de Aria.

Registra cada tarea/intento terminado (objetivo, resultado, motivo, lección) en
tasks/memoria.jsonl con su embedding, y recupera los episodios más parecidos a
un objetivo nuevo para inyectarlos en el system prompt como EXPERIENCIA
RELEVANTE: así Aria recuerda sus errores y aciertos por relevancia en vez de
releerlo todo.

Best-effort en todo (mismo espíritu que core/lecciones.py): sin red o sin cuota
degrada a similitud por palabras, y jamás lanza — la memoria nunca puede
bloquear una tarea.
"""

import json
import logging
import math
import os
import re
from datetime import datetime, timezone

import httpx

from config import GEMINI_API_KEY, GEMINI_BASE, RAG_TOPK

logger = logging.getLogger("aria.memoria")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA = os.path.join(_ROOT, "tasks", "memoria.jsonl")

_EMBED_URL = f"{GEMINI_BASE}/models/gemini-embedding-001:embedContent"
_EMBED_DIM = 256          # sobra para similitud de tareas; jsonl compacto
_MAX_EPISODIOS = 500      # tope del archivo; al superarlo se conservan los 400 últimos
# Umbrales de "parecido de verdad": bajo esto el episodio es ruido, no recuerdo.
_MIN_COSENO = 0.5
_MIN_SOLAPE = 0.1

_RE_PALABRA = re.compile(r"[a-záéíóúñü0-9]{3,}")
# Palabras vacías frecuentes en objetivos: sin esto, "una"/"con" crean solapes
# fantasma entre tareas que no tienen nada que ver.
_STOPWORDS = frozenset(
    "una uno unos unas con por que del los las para como este esta esto sus mas"
    " muy hay son ser fue tras hasta desde entre sobre luego cada".split())


def _embed(texto: str):
    """Vector del texto vía Gemini, o None si falla (sin red, 429, sin key).
    ponytail: llamada one-shot sin rate limiter — el endpoint de embeddings tiene
    cuota aparte de generateContent y el fallback por palabras cubre el fallo."""
    if not (texto and GEMINI_API_KEY):
        return None
    try:
        resp = httpx.post(
            _EMBED_URL,
            json={"model": "models/gemini-embedding-001",
                  "content": {"parts": [{"text": texto[:2000]}]},
                  "outputDimensionality": _EMBED_DIM},
            headers={"x-goog-api-key": GEMINI_API_KEY,
                     "Content-Type": "application/json"},
            timeout=10.0,
        )
        if resp.status_code != 200:
            logger.debug("Embed HTTP %d — fallback a palabras.", resp.status_code)
            return None
        vec = resp.json().get("embedding", {}).get("values")
        return [float(x) for x in vec] if vec else None
    except Exception:                              # noqa: BLE001
        return None


def _coseno(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _palabras(texto: str) -> set:
    return set(_RE_PALABRA.findall((texto or "").lower())) - _STOPWORDS


def _solape(a: str, b: str) -> float:
    """Similitud por palabras (fallback sin embeddings): |A∩B| / |A∪B|."""
    pa, pb = _palabras(a), _palabras(b)
    return len(pa & pb) / len(pa | pb) if pa and pb else 0.0


def cargar() -> list[dict]:
    """Lee los episodios del jsonl. Resiliente: [] si falta; las líneas corruptas
    se saltan (un crash a mitad de escritura no invalida la memoria entera)."""
    episodios: list[dict] = []
    try:
        with open(RUTA, "r", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea:
                    continue
                try:
                    ep = json.loads(linea)
                except json.JSONDecodeError:
                    continue
                if isinstance(ep, dict) and ep.get("objetivo"):
                    episodios.append(ep)
    except OSError:
        pass
    return episodios


def registrar_episodio(objetivo: str, resultado: str, motivo: str = "",
                       leccion: str = "", ciclos: int = 0) -> None:
    """Añade un episodio (best-effort: nunca lanza). Los 429 no se registran —
    quedarse sin cuota no enseña nada sobre la tarea."""
    try:
        if not objetivo or resultado == "limite":
            return
        ep = {
            "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "objetivo": " ".join(objetivo.split())[:300],
            "resultado": resultado,
            "motivo": " ".join((motivo or "").split())[:200],
            "leccion": " ".join((leccion or "").split())[:200],
            "ciclos": int(ciclos),
        }
        ep["vec"] = _embed(f"{ep['objetivo']} {ep['motivo']}")
        episodios = cargar()
        episodios.append(ep)
        if len(episodios) > _MAX_EPISODIOS:
            episodios = episodios[-400:]
        os.makedirs(os.path.dirname(RUTA), exist_ok=True)
        tmp = RUTA + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for e in episodios:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        os.replace(tmp, RUTA)
        logger.info("Episodio registrado (%s): %s", resultado, ep["objetivo"][:60])
    except Exception:                              # noqa: BLE001
        logger.debug("No se pudo registrar el episodio.", exc_info=True)


def relevantes(objetivo: str, k: int | None = None) -> list[dict]:
    """Top-k episodios más parecidos al objetivo. Coseno si hay vector a ambos
    lados; si no, solape de palabras. ponytail: brute-force sobre ≤500 episodios;
    si algún día son miles, ahí sí índice/numpy. Nunca lanza."""
    try:
        k = RAG_TOPK if k is None else k
        if k <= 0 or not objetivo:
            return []
        episodios = cargar()
        if not episodios:
            return []
        qvec = _embed(objetivo)
        puntuados = []
        for ep in episodios:
            if qvec and ep.get("vec"):
                s, minimo = _coseno(qvec, ep["vec"]), _MIN_COSENO
            else:
                s = _solape(objetivo, f"{ep.get('objetivo', '')} {ep.get('motivo', '')}")
                minimo = _MIN_SOLAPE
            if s >= minimo:
                puntuados.append((s, ep))
        puntuados.sort(key=lambda t: t[0], reverse=True)
        return [ep for _, ep in puntuados[:k]]
    except Exception:                              # noqa: BLE001
        return []


def seccion_prompt(objetivo: str) -> str:
    """Sección EXPERIENCIA RELEVANTE para el system prompt ('' si nada útil)."""
    episodios = relevantes(objetivo)
    if not episodios:
        return ""
    lineas = []
    for ep in episodios:
        que = ("la completaste" if ep.get("resultado") == "completado"
               else f"falló por: {ep.get('motivo') or ep.get('resultado')}")
        extra = f" Lección: {ep['leccion']}" if ep.get("leccion") else ""
        lineas.append(f"- «{ep['objetivo'][:80]}» → {que}.{extra}")
    return ("\n\nEXPERIENCIA RELEVANTE — en tareas parecidas a la actual:\n"
            + "\n".join(lineas) + "\n")
