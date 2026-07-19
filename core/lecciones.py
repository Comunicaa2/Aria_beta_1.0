"""core/lecciones.py — Memoria de lecciones de Aria (estilo SOUL.md).

Conocimiento acumulado de errores pasados que se INYECTA en el system prompt en
cada sesión para que Aria no repita los mismos fallos. Gemini no aprende entre
sesiones (sus pesos son fijos); el 'aprendizaje' real vive en este archivo de
texto, tasks/lecciones.json, que se inyecta en el contexto del modelo.

Lo escriben dos actores: el Entrenador (proceso externo, si existe) y la propia
Aria vía `registrar()` al terminar una tarea sin completarla — así el ciclo de
aprendizaje funciona también sin el sistema de entrenamiento.
"""

import json
import logging
import os

logger = logging.getLogger("aria.lecciones")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA = os.path.join(_ROOT, "tasks", "lecciones.json")


def cargar() -> list[dict]:
    """Lee las lecciones de disco. Resiliente: [] si falta o está corrupto."""
    try:
        with open(RUTA, "r", encoding="utf-8") as f:
            datos = json.load(f)
        lecs = datos.get("lecciones", []) if isinstance(datos, dict) else []
        return [x for x in lecs if isinstance(x, dict) and x.get("regla")]
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def registrar(regla: str) -> None:
    """Añade una lección (o suma 1 a 'veces' si ya existe). Escritura atómica.
    Se capa a 40 lecciones (las más frecuentes) para no inflar el prompt."""
    regla = " ".join((regla or "").split())[:200].strip()
    if not regla:
        return
    lecs = cargar()
    for x in lecs:
        if x["regla"].strip().lower() == regla.lower():
            x["veces"] = int(x.get("veces", 0)) + 1
            break
    else:
        lecs.append({"regla": regla, "veces": 1})
    lecs = sorted(lecs, key=lambda x: int(x.get("veces", 0)), reverse=True)[:40]
    try:
        os.makedirs(os.path.dirname(RUTA), exist_ok=True)
        tmp = RUTA + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"lecciones": lecs}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, RUTA)
        logger.info("Lección registrada: %s", regla)
    except OSError as exc:
        logger.warning("No se pudo guardar la lección: %s", exc)


def seccion_prompt() -> str:
    """Sección de lecciones para el system prompt ('' si no hay). Reglas concisas
    (no volcado de logs), ordenadas por frecuencia descendente."""
    lecs = cargar()
    if not lecs:
        return ""
    ordenadas = sorted(lecs, key=lambda x: int(x.get("veces", 0)), reverse=True)
    lineas = "\n".join(f"- {x['regla']}" for x in ordenadas)
    return ("\n\nLECCIONES DE SESIONES ANTERIORES — no repitas estos errores:\n"
            + lineas + "\n")
