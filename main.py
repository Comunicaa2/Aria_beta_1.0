"""
main.py — Punto de entrada y consola de Aria 1.0.

Orquesta el ciclo cognitivo:
    PERCEPCIÓN (captura en RAM) → RAZONAMIENTO (Gemini) → ACTUACIÓN (SO) → repetir.

Características clave:
  · FSM IDLE/WORKING/THINKING/OVERLOADED guía el flujo y anima el avatar.
  · Telemetría: si la CPU/RAM se saturan → OVERLOADED + razonamiento superficial.
  · Parada limpia ante 429: termina la acción, guarda estado y se detiene; al
    reiniciar, ofrece continuar la tarea pendiente.

Uso:
    python main.py

Consola:
    <comando>   Aria intenta cumplir la tarea (hasta MAX_PASOS_TAREA ciclos)
    estado      muestra FSM, avatar y telemetría
    salir       detiene Aria limpiamente
"""

import logging
import sys

from config import (
    AGENT_NAME,
    AGENT_VERSION,
    DELAY_ESTABILIDAD,
    LOG_LEVEL,
    MAX_PASOS_TAREA,
    PAUSA_OVERLOADED,
)
from core.brain import Cerebro, LimiteAPIError
from core.fsm import FSM, Estado
from core import state as estado_persistente
from agent.controller import Controller
from agent.telemetry import Telemetria
from avatar.vts import VTuberAvatar
from utils import image

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("aria.main")

# Resultados posibles de una tarea (gobiernan la consola).
COMPLETADO = "completado"
AGOTADO    = "agotado"
LIMITE     = "limite"
ABORTADO   = "abortado"


class Aria:
    """Orquestador del ciclo cognitivo. Integra cerebro, control, telemetría y avatar."""

    def __init__(self) -> None:
        self.telemetria = Telemetria()
        self.fsm = FSM(on_cambio=self._on_estado)
        self.cerebro = Cerebro()
        self.controller = Controller()
        # El avatar consume telemetría para sus movimientos matemáticos.
        self.avatar = VTuberAvatar(proveedor_telemetria=self.telemetria.leer)

        self.stats = {"tareas": 0, "ciclos": 0, "acciones": 0,
                      "fallos": 0, "limite_api": 0}

        logger.info(
            "%s v%s listo — Físico: %s | Visión: %s | Avatar: %s.",
            AGENT_NAME, AGENT_VERSION,
            "SIMULACIÓN" if self.controller.simulacion else "REAL",
            "ok" if image.disponible() else "inactiva",
            "activo" if self.avatar.activo else "inactivo",
        )

    # ── Listener de la FSM → anima el avatar ────────────────────────────────────
    def _on_estado(self, nuevo: Estado) -> None:
        self.avatar.set_estado(nuevo)

    # ── Ciclo cognitivo de una tarea ────────────────────────────────────────────
    def ejecutar_tarea(self, objetivo: str, ciclo_inicial: int = 1,
                       reanudada: bool = False) -> str:
        """
        Ejecuta una tarea hasta completarla, agotar los ciclos o tocar el límite
        de API. Devuelve uno de: COMPLETADO / AGOTADO / LIMITE / ABORTADO.
        """
        if not reanudada:
            self.cerebro.reset()
        self.stats["tareas"] += 1
        fallos_seguidos = 0

        try:
            for ciclo in range(ciclo_inicial, MAX_PASOS_TAREA + 1):
                self.stats["ciclos"] += 1

                # ── Telemetría → ¿OVERLOADED? ──────────────────────────────
                lec = self.telemetria.leer()
                sobrecargado = lec.sobrecargado
                if sobrecargado:
                    self.fsm.a_overloaded()
                    logger.warning("OVERLOADED — %s. Pausa %.1fs y razonamiento superficial.",
                                   lec.resumen(), PAUSA_OVERLOADED)
                    self._dormir(PAUSA_OVERLOADED)

                # Profundidad de razonamiento: profundo al arrancar o si está
                # atascada; superficial si el PC va saturado (máxima velocidad).
                profundo = (not sobrecargado) and (ciclo_inicial == ciclo or fallos_seguidos >= 2)

                # ── 1. PERCEPCIÓN (en RAM; consola fuera de la captura) ────
                self.fsm.a_thinking()
                image.minimizar_consola()
                cap = image.capturar()
                if cap is None or not cap.b64:
                    logger.error("Percepción fallida: sin captura — no se actúa a ciegas.")
                    return ABORTADO

                # ── 2. RAZONAMIENTO (Gemini: imagen + comando) ─────────────
                texto = (
                    f"Tarea: {objetivo}\n"
                    f"Ciclo {ciclo}/{MAX_PASOS_TAREA}. "
                    f"La imagen mide {cap.ancho_img}x{cap.alto_img} px "
                    f"(usa esas coordenadas). ¿Cuál es la siguiente acción?"
                )
                decision = self.cerebro.pensar(texto, cap.b64, profundo=profundo)

                if not decision.valida:
                    fallos_seguidos += 1
                    self.stats["fallos"] += 1
                    logger.warning("Sin ACCION válida (fallo %d). Reintentando.", fallos_seguidos)
                    self.cerebro.registrar_resultado(
                        "Sistema: tu respuesta no contenía una ACCION válida. "
                        "Responde EXACTAMENTE con PENSAMIENTO / ACCION / FIN."
                    )
                    if fallos_seguidos >= 3:
                        logger.error("Demasiadas respuestas inválidas — abortando tarea.")
                        return ABORTADO
                    continue

                if decision.pensamiento:
                    logger.info("💭 %s", decision.pensamiento[:120])

                # ── 3. ACTUACIÓN ───────────────────────────────────────────
                self.fsm.a_working()
                logger.info("▶ ACCION: %s", decision.accion)
                ok = self.controller.ejecutar(decision.accion, cap)
                self.stats["acciones"] += 1

                if self.controller.es_done:
                    logger.info("Tarea declarada COMPLETA por el modelo (ciclo %d).", ciclo)
                    return COMPLETADO

                if ok:
                    fallos_seguidos = 0
                    image.esperar_estabilidad(max_seg=DELAY_ESTABILIDAD + 1.5)
                    self.cerebro.registrar_resultado(
                        f"Sistema: acción '{decision.accion}' ejecutada. Evalúa la nueva captura."
                    )
                else:
                    fallos_seguidos += 1
                    self.stats["fallos"] += 1
                    self.cerebro.registrar_resultado(
                        f"Sistema: la acción '{decision.accion}' falló o no es válida. Prueba otra."
                    )

            logger.info("Se alcanzó el tope de %d ciclos.", MAX_PASOS_TAREA)
            return AGOTADO

        except LimiteAPIError:
            # ── PARADA LIMPIA: guardar estado para reanudar tras el límite ──
            self.stats["limite_api"] += 1
            logger.warning("Límite de API (429) alcanzado — guardando estado y deteniendo.")
            self._guardar_pendiente(objetivo, ciclo)
            return LIMITE
        finally:
            image.restaurar_consola()
            self.fsm.a_idle()

    # ── Persistencia ────────────────────────────────────────────────────────────
    def _guardar_pendiente(self, objetivo: str, ciclo: int) -> None:
        snap = estado_persistente.EstadoGuardado(
            objetivo=objetivo,
            ciclo=ciclo,
            completado=False,
            fsm=self.fsm.estado.name,
            historial=self.cerebro.exportar_historial(),
            stats=self.stats,
        )
        estado_persistente.guardar(snap)

    def cargar_pendiente(self):
        """Devuelve el EstadoGuardado pendiente (o None). Restaura stats e historial."""
        snap = estado_persistente.cargar()
        if snap is None:
            return None
        if snap.stats:
            self.stats.update(snap.stats)
        if snap.hay_pendiente:
            self.cerebro.importar_historial(snap.historial)
            return snap
        return None

    # ── Util ─────────────────────────────────────────────────────────────────────
    def _dormir(self, segundos: float) -> None:
        import time
        time.sleep(max(0.0, segundos))

    def estado_resumen(self) -> str:
        return (f"FSM: {self.fsm.estado.name} | Avatar: "
                f"{'activo' if self.avatar.activo else 'inactivo'} | "
                f"{self.telemetria.leer().resumen()} | stats={self.stats}")

    def detener(self) -> None:
        self.avatar.cerrar()
        self.cerebro.cerrar()
        self.fsm.a_idle()
        logger.info("%s detenido.", AGENT_NAME)


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLA
# ══════════════════════════════════════════════════════════════════════════════

def _banner(aria: Aria) -> None:
    print(
        f"\n{'═' * 64}\n"
        f"  {AGENT_NAME} {AGENT_VERSION} — Agente autónomo de PC (Gemini 2.5 Flash)\n"
        f"  Avatar: {'ACTIVO' if aria.avatar.activo else 'inactivo'} | "
        f"Físico: {'SIMULACIÓN' if aria.controller.simulacion else 'REAL'}\n"
        f"  Escribe una tarea. 'estado' para ver telemetría. 'salir' para cerrar.\n"
        f"{'═' * 64}\n"
    )


def _resolver_pendiente(aria: Aria) -> None:
    """Si hay tarea pendiente de un 429 previo, ofrece continuarla."""
    snap = aria.cargar_pendiente()
    if snap is None:
        return
    print(f"\n  ⏸ Tarea pendiente del cierre anterior: «{snap.objetivo}» "
          f"(ciclo {snap.ciclo}).")
    try:
        resp = input("  ¿Continuar esa tarea? [s/N] > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        resp = "n"
    if resp in ("s", "si", "sí", "y", "yes"):
        res = aria.ejecutar_tarea(snap.objetivo, ciclo_inicial=snap.ciclo, reanudada=True)
        _reportar(res)
        if res != LIMITE:
            estado_persistente.limpiar()
    else:
        estado_persistente.limpiar()
        print("  (Tarea pendiente descartada.)")


def _reportar(resultado: str) -> None:
    msg = {
        COMPLETADO: "✓ Tarea completada.",
        AGOTADO:    "⚠ Se agotaron los ciclos sin declarar fin.",
        LIMITE:     "⏸ Límite de API (429): estado guardado. Reinicia para continuar.",
        ABORTADO:   "✗ Tarea abortada (percepción o respuestas inválidas).",
    }.get(resultado, resultado)
    print(f"\n  {msg}\n")


def main() -> None:
    aria = Aria()
    _banner(aria)
    _resolver_pendiente(aria)

    try:
        while True:
            try:
                entrada = input("Tarea > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not entrada:
                continue
            low = entrada.lower()
            if low in ("salir", "exit", "quit", "q"):
                break
            if low in ("estado", "status"):
                print(f"  → {aria.estado_resumen()}")
                continue

            resultado = aria.ejecutar_tarea(entrada)
            _reportar(resultado)

            if resultado == LIMITE:
                print("  Deteniendo por límite de API. Vuelve a ejecutar para reanudar.")
                break
            if resultado == COMPLETADO:
                estado_persistente.limpiar()
    finally:
        aria.detener()
        print(f"\n{AGENT_NAME}: hasta la próxima. ✦\n")


if __name__ == "__main__":
    main()
