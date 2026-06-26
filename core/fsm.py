"""
core/fsm.py — Máquina de estados finita (FSM) de Aria 1.0.

Estados:
    IDLE        Reposo. Cero gasto de API; esperando comando.
    WORKING     Ejecutando acciones físicas en el SO.
    THINKING    Razonando (llamada a Gemini con presupuesto de pensamiento).
    OVERLOADED  CPU/RAM/temperatura saturados → reducir profundidad de razonamiento.

La FSM es thread-safe y notifica cada transición a un listener opcional (el avatar
VTuber). El orquestador consulta `estado` y usa los helpers (`a_idle`, etc.) para
transicionar de forma legible.
"""

import logging
import threading
from enum import Enum, auto
from typing import Callable, Optional

logger = logging.getLogger("aria.fsm")


class Estado(Enum):
    IDLE       = auto()
    WORKING    = auto()
    THINKING   = auto()
    OVERLOADED = auto()


class FSM:
    """Máquina de estados thread-safe con listener de transiciones."""

    def __init__(self, on_cambio: Optional[Callable[[Estado], None]] = None) -> None:
        self._estado = Estado.IDLE
        self._lock = threading.Lock()
        self._on_cambio = on_cambio

    @property
    def estado(self) -> Estado:
        with self._lock:
            return self._estado

    @property
    def ocupada(self) -> bool:
        return self.estado != Estado.IDLE

    def set(self, nuevo: Estado) -> None:
        """Transiciona al nuevo estado y notifica al listener (fuera del lock)."""
        with self._lock:
            if nuevo == self._estado:
                return
            anterior = self._estado
            self._estado = nuevo
        logger.info("Estado: %s → %s.", anterior.name, nuevo.name)
        if self._on_cambio is not None:
            try:
                self._on_cambio(nuevo)
            except Exception as exc:               # noqa: BLE001
                # El avatar nunca debe tumbar el ciclo cognitivo.
                logger.debug("FSM listener falló (%s) — ignorado.", exc)

    # ── Helpers legibles ────────────────────────────────────────────────────────
    def a_idle(self) -> None:        self.set(Estado.IDLE)
    def a_working(self) -> None:     self.set(Estado.WORKING)
    def a_thinking(self) -> None:    self.set(Estado.THINKING)
    def a_overloaded(self) -> None:  self.set(Estado.OVERLOADED)
