"""
core/state.py — Guardado y carga de estado de Aria 1.0 (parada limpia).

Cuando se alcanza el límite de la API (429) Aria termina la acción en curso y
GUARDA: la tarea pendiente, el historial de conversación, el estado de la FSM y
las estadísticas. Al reiniciar, `cargar()` recupera ese estado para continuar la
tarea donde se quedó.

Formato: un único JSON en disco (STATE_FILE). Escritura atómica (archivo temporal
+ replace) para no corromper el estado si el proceso muere a mitad de guardado.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from config import STATE_FILE

logger = logging.getLogger("aria.state")


@dataclass
class EstadoGuardado:
    """Snapshot persistente de Aria."""
    objetivo: str = ""                       # tarea pendiente ("" = ninguna)
    ciclo: int = 0                           # ciclo de visión alcanzado
    completado: bool = True                  # True si no quedó nada pendiente
    fsm: str = "IDLE"                        # último estado de la FSM
    historial: list = field(default_factory=list)   # turnos [{role, content}]
    stats: dict = field(default_factory=dict)        # contadores acumulados
    guardado_en: float = 0.0                 # timestamp epoch del guardado

    @property
    def hay_pendiente(self) -> bool:
        return bool(self.objetivo) and not self.completado


def guardar(estado: EstadoGuardado) -> bool:
    """Guarda el estado en disco de forma atómica. Resiliente: nunca lanza."""
    estado.guardado_en = time.time()
    tmp = STATE_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(estado), f, ensure_ascii=False, indent=2)
        os.replace(tmp, STATE_FILE)             # reemplazo atómico
        logger.info("Estado guardado en '%s' (pendiente=%s).",
                    STATE_FILE, estado.hay_pendiente)
        return True
    except Exception as exc:                     # noqa: BLE001
        logger.warning("No se pudo guardar el estado: %s", exc)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return False


def cargar() -> Optional[EstadoGuardado]:
    """Carga el estado guardado, o None si no existe / está corrupto."""
    if not os.path.isfile(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            datos = json.load(f)
        # Filtra solo las claves conocidas (tolera versiones futuras del archivo).
        validas = {k: datos[k] for k in EstadoGuardado().__dict__ if k in datos}
        estado = EstadoGuardado(**validas)
        logger.info("Estado cargado de '%s' (pendiente=%s).",
                    STATE_FILE, estado.hay_pendiente)
        return estado
    except Exception as exc:                     # noqa: BLE001
        logger.warning("No se pudo cargar el estado ('%s') — se ignora: %s",
                       STATE_FILE, exc)
        return None


def limpiar() -> None:
    """Borra el archivo de estado (al completar limpiamente una tarea)."""
    try:
        if os.path.isfile(STATE_FILE):
            os.remove(STATE_FILE)
            logger.debug("Archivo de estado eliminado.")
    except OSError as exc:
        logger.debug("No se pudo eliminar el estado: %s", exc)
