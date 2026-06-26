"""
avatar/vts.py — Avatar VTuber de Aria 1.0 (VTube Studio WebSocket). OPCIONAL.

Dos capas, ambas en un hilo propio para NO bloquear el ciclo cognitivo:

  1. ESTADO DE LA FSM → un Hotkey persistente (expresión):
        THINKING   → "aria_pensando"
        WORKING    → "aria_concentrada"
        OVERLOADED → "aria_panico"
        IDLE       → (limpia la expresión activa; cara neutra)

  2. MOVIMIENTO MATEMÁTICO basado en TELEMETRÍA del PC:
        La cabeza cabecea con una onda senoidal cuya amplitud y frecuencia
        escalan con la CARGA de CPU; la "respiración" (posición vertical) sigue
        el uso de RAM. Se inyecta vía InjectParameterDataRequest (sin webcam).

CREA ESTOS HOTKEYS EN VTUBE STUDIO (Settings → Hotkeys, tipo "Toggle Expression"):
    aria_pensando    aria_concentrada    aria_panico

REGLA DE DEGRADACIÓN: si falta websocket-client o VTube Studio está cerrado,
`activo` queda en False y TODOS los métodos son no-op silenciosos. Nunca crashea.
"""

import json
import logging
import math
import os
import threading
import time
import uuid
from typing import Callable, Optional

from config import (
    VTUBE_ENABLED,
    VTUBE_PLUGIN_DEV,
    VTUBE_PLUGIN_NAME,
    VTUBE_TIMEOUT_APROBAR,
    VTUBE_TIMEOUT_CONECTAR,
    VTUBE_TIMEOUT_RECIBIR,
    VTUBE_TOKEN_FILE,
    VTUBE_WS_URL,
)
from core.fsm import Estado

logger = logging.getLogger("aria.avatar")

try:
    import websocket as _ws_lib
    _WS_OK = True
except ImportError:
    _ws_lib = None                                 # type: ignore[assignment]
    _WS_OK = False

# Estado de la FSM → hotkeyID (None = limpiar / neutro).
_ESTADO_HOTKEY: dict[Estado, Optional[str]] = {
    Estado.THINKING:   "aria_pensando",
    Estado.WORKING:    "aria_concentrada",
    Estado.OVERLOADED: "aria_panico",
    Estado.IDLE:       None,
}

_TICK_ANIM = 1 / 15.0    # 15 Hz de animación matemática


class VTuberAvatar:
    """Puente resiliente con VTube Studio. Animación en hilo propio."""

    def __init__(self, proveedor_telemetria: Optional[Callable[[], object]] = None) -> None:
        self.activo = False
        self._ws = None
        self._lock = threading.Lock()
        self._proveedor = proveedor_telemetria     # devuelve una Lectura (cpu/ram)
        self._hotkey_activo: Optional[str] = None
        self._stop = threading.Event()
        self._hilo: Optional[threading.Thread] = None

        if not VTUBE_ENABLED:
            logger.info("Avatar deshabilitado en config (VTUBE_ENABLED=False).")
            return
        if not _WS_OK:
            logger.warning("Avatar inactivo: falta 'websocket-client' "
                           "(pip install websocket-client).")
            return

        try:
            self._conectar()
        except Exception as exc:                   # noqa: BLE001
            self.activo = False
            self._ws = None
            logger.warning("Avatar inactivo: continuando sin VTuber (%s).", exc)

        if self.activo:
            self._hilo = threading.Thread(target=self._bucle_anim,
                                          name="AriaAvatar", daemon=True)
            self._hilo.start()

    # ── API pública ────────────────────────────────────────────────────────────
    def set_estado(self, estado: Estado) -> None:
        """Refleja el estado de la FSM como expresión persistente. No-op si inactivo."""
        if not self.activo:
            return
        hotkey = _ESTADO_HOTKEY.get(estado)
        if hotkey == self._hotkey_activo:
            return
        try:
            if self._hotkey_activo:                # apaga la expresión anterior (toggle)
                self._disparar_hotkey(self._hotkey_activo)
            if hotkey:
                self._disparar_hotkey(hotkey)
            self._hotkey_activo = hotkey
        except Exception as exc:                   # noqa: BLE001
            logger.debug("Avatar set_estado falló (%s) — ignorado.", exc)

    def cerrar(self) -> None:
        """Detiene la animación y cierra la conexión. Resiliente."""
        self._stop.set()
        if self._hilo and self._hilo.is_alive():
            self._hilo.join(timeout=2.0)
        if not self.activo:
            return
        try:
            if self._hotkey_activo:
                self._disparar_hotkey(self._hotkey_activo)   # vuelve a neutro
            with self._lock:
                if self._ws is not None:
                    self._ws.close()
        except Exception:                          # noqa: BLE001
            pass
        finally:
            self.activo = False
            self._ws = None
            logger.info("Avatar desconectado.")

    # ── Animación matemática (telemetría → parámetros del modelo) ──────────────
    def _bucle_anim(self) -> None:
        t0 = time.monotonic()
        while not self._stop.is_set():
            try:
                cpu, ram = self._telemetria()
                t = time.monotonic() - t0
                carga = max(0.0, min(cpu / 100.0, 1.0))
                mem = max(0.0, min(ram / 100.0, 1.0))

                # Cabeceo: amplitud y frecuencia crecen con la CPU.
                freq = 0.5 + carga * 2.2
                amp = 5.0 + carga * 18.0
                ang_x = amp * math.sin(2 * math.pi * freq * t)
                ang_y = amp * 0.6 * math.sin(2 * math.pi * freq * 0.7 * t + 1.0)
                # Respiración: la RAM modula la amplitud vertical.
                resp = (0.4 + mem * 0.6) * math.sin(2 * math.pi * 0.25 * t)

                self._inyectar({
                    "FaceAngleX": ang_x,
                    "FaceAngleY": ang_y,
                    "FaceAngleZ": ang_x * 0.3,
                    "FacePositionY": resp,
                })
            except Exception as exc:               # noqa: BLE001
                logger.debug("Avatar anim: fallo (%s) — sigo.", exc)
            self._stop.wait(_TICK_ANIM)

    def _telemetria(self) -> tuple[float, float]:
        if self._proveedor is None:
            return 0.0, 0.0
        try:
            lec = self._proveedor()
            return float(getattr(lec, "cpu_pct", 0.0)), float(getattr(lec, "ram_pct", 0.0))
        except Exception:                          # noqa: BLE001
            return 0.0, 0.0

    def _inyectar(self, valores: dict) -> None:
        payload = _payload("InjectParameterDataRequest", {
            "faceFound": False,
            "mode": "set",
            "parameterValues": [{"id": k, "value": v} for k, v in valores.items()],
        })
        self._enviar_recibir(payload)

    # ── Conexión / autenticación ────────────────────────────────────────────────
    def _conectar(self) -> None:
        logger.info("Avatar: conectando a %s…", VTUBE_WS_URL)
        ws = _ws_lib.create_connection(VTUBE_WS_URL, timeout=VTUBE_TIMEOUT_CONECTAR)
        ws.settimeout(VTUBE_TIMEOUT_RECIBIR)
        self._ws = ws
        self._autenticar()

    def _autenticar(self) -> None:
        token = _cargar_token()
        if token and self._auth_token(token):
            self.activo = True
            logger.info("Avatar autenticado (token guardado). ✦")
            return
        nuevo = self._pedir_token()
        if not nuevo:
            raise RuntimeError("VTube Studio no entregó token (rechazado o timeout).")
        _guardar_token(nuevo)
        if self._auth_token(nuevo):
            self.activo = True
            logger.info("Avatar autenticado (token nuevo). ✦")
        else:
            raise RuntimeError("VTube Studio rechazó el nuevo token.")

    def _pedir_token(self) -> Optional[str]:
        payload = _payload("AuthenticationTokenRequest", {
            "pluginName": VTUBE_PLUGIN_NAME,
            "pluginDeveloper": VTUBE_PLUGIN_DEV,
        })
        logger.info("Avatar: aprueba el plugin en VTube Studio (%ds)…", VTUBE_TIMEOUT_APROBAR)
        with self._lock:
            self._ws.send(json.dumps(payload))
            self._ws.settimeout(VTUBE_TIMEOUT_APROBAR)
            try:
                raw = self._ws.recv()
            finally:
                self._ws.settimeout(VTUBE_TIMEOUT_RECIBIR)
        token = json.loads(raw).get("data", {}).get("authenticationToken", "").strip()
        return token or None

    def _auth_token(self, token: str) -> bool:
        payload = _payload("AuthenticationRequest", {
            "pluginName": VTUBE_PLUGIN_NAME,
            "pluginDeveloper": VTUBE_PLUGIN_DEV,
            "authenticationToken": token,
        })
        try:
            return self._enviar_recibir(payload).get("data", {}).get("authenticated", False)
        except Exception:                          # noqa: BLE001
            return False

    # ── Helpers ─────────────────────────────────────────────────────────────────
    def _disparar_hotkey(self, hotkey_id: str) -> None:
        self._enviar_recibir(_payload("HotkeyTriggerRequest", {"hotkeyID": hotkey_id}))
        logger.debug("Avatar hotkey → '%s'.", hotkey_id)

    def _enviar_recibir(self, payload: dict) -> dict:
        with self._lock:
            self._ws.send(json.dumps(payload))
            raw = self._ws.recv()
        return json.loads(raw)


# ─── Funciones de módulo ──────────────────────────────────────────────────────
def _payload(tipo: str, datos: dict) -> dict:
    return {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": str(uuid.uuid4()),
        "messageType": tipo,
        "data": datos,
    }


def _cargar_token() -> Optional[str]:
    try:
        if os.path.isfile(VTUBE_TOKEN_FILE):
            with open(VTUBE_TOKEN_FILE, "r", encoding="utf-8") as f:
                return f.read().strip() or None
    except OSError:
        pass
    return None


def _guardar_token(token: str) -> None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(VTUBE_TOKEN_FILE)), exist_ok=True)
        with open(VTUBE_TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write(token)
        logger.info("Token VTube Studio guardado.")
    except OSError as exc:
        logger.warning("No se pudo guardar el token VTuber: %s.", exc)
