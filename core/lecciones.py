"""core/lecciones.py — Memoria de lecciones de Aria (estilo SOUL.md).

Conocimiento acumulado de errores pasados que se INYECTA en el system prompt en
cada sesión para que Aria no repita los mismos fallos. Gemini no aprende entre
sesiones (sus pesos son fijos); el 'aprendizaje' real vive en este archivo de
texto, tasks/lecciones.json, que se inyecta en el contexto del modelo.

Aria SOLO LEE este archivo. El Entrenador (proceso externo) es quien lo
escribe/actualiza, respetando el desacople: la comunicación es por el archivo,
no por código interno de Aria.
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
