"""core/skills.py — Biblioteca de skills de Aria (estilo Voyager/Hermes Agent).

Una skill es un script `skill_*.py` en WORKSPACE_DIR que la propia Aria escribió
con la acción 'guardar' y puede reutilizar con 'ejecutar_python'. La primera
línea de su docstring es la descripción. El catálogo se inyecta en el system
prompt al inicio de cada tarea: el conocimiento operativo crece entre tareas y
sesiones — Aria se optimiza a sí misma escribiendo y refinando sus skills.
"""

import ast
import glob
import logging
import os

from config import WORKSPACE_DIR

logger = logging.getLogger("aria.skills")


def listar() -> list[tuple[str, str]]:
    """[(nombre, descripción)] de las skills en workspace/. Resiliente: una
    skill con sintaxis rota se lista igual (Aria puede corregirla)."""
    skills = []
    for ruta in sorted(glob.glob(os.path.join(WORKSPACE_DIR, "skill_*.py"))):
        try:
            with open(ruta, "r", encoding="utf-8", errors="replace") as f:
                doc = ast.get_docstring(ast.parse(f.read())) or ""
        except (OSError, SyntaxError, ValueError):
            doc = "(no se pudo leer la descripción)"
        desc = " ".join(doc.split())[:120] or "(sin descripción)"
        skills.append((os.path.basename(ruta), desc))
    return skills


def seccion_prompt() -> str:
    """Sección de skills para el system prompt ('' si no hay ninguna)."""
    skills = listar()
    if not skills:
        return ""
    lineas = "\n".join(f"- {n}: {d}" for n, d in skills)
    return ("\n\nSKILLS DISPONIBLES — scripts tuyos ya probados; reutilízalos con "
            "ejecutar_python NOMBRE [argumentos] en vez de rehacer el trabajo:\n"
            + lineas + "\n")
