"""core/skills.py — Biblioteca de skills de Aria (estilo Voyager/Hermes Agent).

Dos orígenes, un solo catálogo:
  · skills/ (repo)    — biblioteca BASE por categorías (sistema, red, archivos,
                        ventanas, datos…), versionada y compartida.
  · workspace/        — skills que la propia Aria escribió con 'guardar'.

Una skill es un script `skill_*.py`; la primera línea de su docstring es la
descripción (y debe decir qué argumentos recibe). Si un nombre existe en ambos
orígenes gana el de workspace/: Aria puede mejorar una skill de fábrica sin
tocar el repo. El catálogo se inyecta en el system prompt al inicio de cada
tarea: el conocimiento operativo crece entre tareas y sesiones.
"""

import ast
import glob
import logging
import os

from config import SKILLS_DIR, WORKSPACE_DIR

logger = logging.getLogger("aria.skills")


def _describir(ruta: str) -> str:
    try:
        with open(ruta, "r", encoding="utf-8", errors="replace") as f:
            doc = ast.get_docstring(ast.parse(f.read())) or ""
    except (OSError, SyntaxError, ValueError):
        doc = "(no se pudo leer la descripción)"
    return " ".join(doc.split())[:140] or "(sin descripción)"


def listar() -> list[tuple[str, str]]:
    """[(nombre, descripción)] del catálogo completo. Workspace pisa a la base."""
    encontrados: dict[str, str] = {}
    for base in (SKILLS_DIR, WORKSPACE_DIR):      # workspace al final → gana
        for ruta in glob.glob(os.path.join(base, "skill_*.py")):
            encontrados[os.path.basename(ruta)] = _describir(ruta)
    return sorted(encontrados.items())


def seccion_prompt() -> str:
    """Sección de skills para el system prompt ('' si no hay ninguna)."""
    skills = listar()
    if not skills:
        return ""
    lineas = "\n".join(f"- {n}: {d}" for n, d in skills)
    return ("\n\nSKILLS DISPONIBLES — scripts ya probados; reutilízalos con "
            "ejecutar_python NOMBRE [argumentos] en vez de rehacer el trabajo:\n"
            + lineas + "\n")
