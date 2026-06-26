"""
agent/telemetry.py — Telemetría de hardware para Aria 1.0.

Lee CPU, RAM y (si hay sensor) temperatura para que la FSM decida cuándo pasar a
OVERLOADED y reducir la profundidad de razonamiento. Usa psutil si está; para la
RAM cae a ctypes (GlobalMemoryStatusEx) en Windows como respaldo.

La lectura de CPU es NO bloqueante: psutil.cpu_percent(interval=None) devuelve el
% desde la última llamada, así no añade latencia al ciclo cognitivo. La primera
lectura tras arrancar puede dar 0.0 (normal: aún no hay intervalo de referencia).

El avatar también consume esta telemetría para sus movimientos matemáticos.
"""

import ctypes
import logging
from dataclasses import dataclass
from typing import Optional

from config import CPU_OVERLOAD_PCT, RAM_OVERLOAD_PCT, TEMP_OVERLOAD_C

logger = logging.getLogger("aria.telemetry")

try:
    import psutil
    _PSUTIL_OK = True
except Exception as _exc:                         # noqa: BLE001
    psutil = None                                 # type: ignore[assignment]
    _PSUTIL_OK = False
    logger.warning("telemetry: psutil no disponible (%s) → fallback ctypes (solo RAM).", _exc)


@dataclass
class Lectura:
    """Instantánea de telemetría. `temp_c` es None si no hay sensor accesible."""
    cpu_pct: float
    ram_pct: float
    ram_libre_mb: float
    temp_c: Optional[float]

    @property
    def sobrecargado(self) -> bool:
        """True si CPU, RAM o temperatura superan sus umbrales de OVERLOADED."""
        if self.cpu_pct >= CPU_OVERLOAD_PCT:
            return True
        if self.ram_pct >= RAM_OVERLOAD_PCT:
            return True
        if self.temp_c is not None and self.temp_c >= TEMP_OVERLOAD_C:
            return True
        return False

    def resumen(self) -> str:
        t = f" | temp {self.temp_c:.0f}°C" if self.temp_c is not None else ""
        return f"CPU {self.cpu_pct:.0f}% | RAM {self.ram_pct:.0f}% ({self.ram_libre_mb:.0f} MB libres){t}"


class Telemetria:
    """Lector de telemetría de PC. `leer()` nunca lanza; degrada a valores seguros."""

    def __init__(self) -> None:
        # Ceba el medidor de CPU de psutil para que la 1.ª lectura real sea útil.
        if _PSUTIL_OK:
            try:
                psutil.cpu_percent(interval=None)
            except Exception:                     # noqa: BLE001
                pass
        logger.info("Telemetría iniciada (%s).", "psutil" if _PSUTIL_OK else "ctypes RAM")

    def leer(self) -> Lectura:
        """Devuelve una `Lectura` actual. No bloquea (cpu_percent sin intervalo)."""
        cpu = self._cpu()
        ram_pct, ram_libre = self._ram()
        temp = self._temp()
        return Lectura(cpu_pct=cpu, ram_pct=ram_pct, ram_libre_mb=ram_libre, temp_c=temp)

    # ── CPU ───────────────────────────────────────────────────────────────────
    @staticmethod
    def _cpu() -> float:
        if not _PSUTIL_OK:
            return 0.0
        try:
            return float(psutil.cpu_percent(interval=None))
        except Exception:                         # noqa: BLE001
            return 0.0

    # ── RAM (psutil → ctypes) ──────────────────────────────────────────────────
    @staticmethod
    def _ram() -> tuple[float, float]:
        if _PSUTIL_OK:
            try:
                vm = psutil.virtual_memory()
                return float(vm.percent), vm.available / (1024 * 1024)
            except Exception:                     # noqa: BLE001
                pass
        return Telemetria._ram_ctypes()

    @staticmethod
    def _ram_ctypes() -> tuple[float, float]:
        """Lee el % de RAM en Windows sin psutil (GlobalMemoryStatusEx)."""
        try:
            class _MEMSTATUS(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = _MEMSTATUS()
            stat.dwLength = ctypes.sizeof(_MEMSTATUS)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return 0.0, 0.0
            return float(stat.dwMemoryLoad), stat.ullAvailPhys / (1024 * 1024)
        except Exception:                         # noqa: BLE001
            return 0.0, 0.0

    # ── Temperatura (mejor esfuerzo; suele faltar en Windows) ──────────────────
    @staticmethod
    def _temp() -> Optional[float]:
        if not _PSUTIL_OK or not hasattr(psutil, "sensors_temperatures"):
            return None
        try:
            sensores = psutil.sensors_temperatures()
            if not sensores:
                return None
            # Toma la temperatura actual más alta entre todos los sensores.
            picos = [
                lectura.current
                for grupo in sensores.values()
                for lectura in grupo
                if lectura.current is not None
            ]
            return max(picos) if picos else None
        except Exception:                         # noqa: BLE001
            return None
