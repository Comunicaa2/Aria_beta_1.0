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
    python main.py             consola manual normal
    python main.py --trainer   además, consume tareas del Entrenador (tasks/active.json)

Consola:
    <comando>   Aria intenta cumplir la tarea (hasta MAX_PASOS_TAREA ciclos)
    estado      muestra FSM, avatar y telemetría
    salir       detiene Aria limpiamente
"""

import logging
import sys
import threading
import time

from config import (
    AGENT_NAME,
    AGENT_VERSION,
    AHORRO_IMG_SIZE,
    AHORRO_JPEG_Q,
    DELAY_ESTABILIDAD,
    LOG_LEVEL,
    MAX_PASOS_TAREA,
    PAUSA_OVERLOADED,
)
from core.brain import (Cerebro, LimiteAPIError,
                        MODELO_GEMINI, MODELO_NIM_LLAMA, MODELO_NIM_OMNI)
from core.fsm import FSM, Estado
from core import state as estado_persistente
from agent.controller import Controller
from agent.telemetry import Telemetria
from avatar.vts import VTuberAvatar
from utils import image

# Modo Entrenador (opcional): import PROTEGIDO. Si trainer/ no existe o falla,
# Aria sigue funcionando con normalidad en modo manual. No rompe nada.
try:
    from trainer import protocolo as _proto
    _TRAINER_OK = True
except Exception:                                  # noqa: BLE001
    _proto = None
    _TRAINER_OK = False

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

        # Serializa la ejecución física: una sola tarea controla el SO a la vez,
        # venga del usuario (consola) o del Entrenador (consumidor). En modo
        # normal el lock está siempre libre, así que no afecta nada.
        self._exec_lock = threading.Lock()
        self._limite_event = threading.Event()     # se activa si Aria toca 429

        # Acciones de la ÚLTIMA tarea repartidas por modelo (para el Entrenador).
        self.ultimas_acciones_modelo = {MODELO_GEMINI: 0,
                                        MODELO_NIM_LLAMA: 0, MODELO_NIM_OMNI: 0}

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
        # Reparto de acciones por modelo para ESTA tarea (lo lee el Entrenador).
        self.ultimas_acciones_modelo = {MODELO_GEMINI: 0,
                                        MODELO_NIM_LLAMA: 0, MODELO_NIM_OMNI: 0}

        try:
            for ciclo in range(ciclo_inicial, MAX_PASOS_TAREA + 1):
                self.stats["ciclos"] += 1

                # ── Telemetría → ¿OVERLOADED? ──────────────────────────────
                lec = self.telemetria.leer()
                sobrecargado = lec.sobrecargado
                if sobrecargado:
                    self.fsm.a_overloaded()
                    logger.warning("OVERLOADED — %s. Modo AHORRO: pausa %.1fs, captura "
                                   "reducida y razonamiento superficial.",
                                   lec.resumen(), PAUSA_OVERLOADED)
                    self._dormir(PAUSA_OVERLOADED)

                # Profundidad de razonamiento: profundo al arrancar o si está
                # atascada; superficial si el PC va saturado (máxima velocidad).
                profundo = (not sobrecargado) and (ciclo_inicial == ciclo or fallos_seguidos >= 2)

                # ── 1. PERCEPCIÓN (en RAM; consola fuera de la captura) ────
                # OVERLOADED mantiene su estado en la FSM; a_thinking solo se usa
                # con el PC sano. En modo AHORRO la captura va reducida.
                if not sobrecargado:
                    self.fsm.a_thinking()
                image.minimizar_consola()
                cap = (image.capturar(max_size=AHORRO_IMG_SIZE, calidad=AHORRO_JPEG_Q)
                       if sobrecargado else image.capturar())
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
                # Indicador visible del modelo que decidió esta acción.
                logger.info("[%s] ▶ ACCION: %s", decision.modelo, decision.accion)
                ok = self.controller.ejecutar(decision.accion, cap)
                self.stats["acciones"] += 1

                if self.controller.es_done:
                    # FIX #3: no aceptar 'done' a ciegas — recapturar y confirmar.
                    cap2 = image.capturar()
                    b64 = cap2.b64 if (cap2 and cap2.b64) else cap.b64
                    if self.cerebro.confirmar_done(objetivo, b64):
                        logger.info("Tarea COMPLETA confirmada (ciclo %d).", ciclo)
                        return COMPLETADO
                    logger.warning("'done' RECHAZADO: la tarea no se ve completa — continúo.")
                    self.cerebro.registrar_resultado(
                        "Sistema: declaraste 'done' pero la verificación visual dice que la "
                        "tarea NO está completa. Sigue trabajando en ella.")
                    fallos_seguidos += 1
                    continue

                # Detalle opcional del controller (p. ej. coords que reportó find_text).
                detalle = (f" {self.controller.ultimo_detalle}."
                           if self.controller.ultimo_detalle else "")
                if ok:
                    fallos_seguidos = 0
                    # Atribuye la acción ejecutada al modelo que la produjo.
                    self.ultimas_acciones_modelo[decision.modelo] = (
                        self.ultimas_acciones_modelo.get(decision.modelo, 0) + 1
                    )
                    # FIX #1: las apps tardan en pintar → piso mayor tras launch_app.
                    es_app = decision.accion.lower().startswith("launch_app")
                    image.esperar_estabilidad(
                        max_seg=DELAY_ESTABILIDAD + 1.5,
                        min_espera=1.5 if es_app else 0.5,
                    )
                    self.cerebro.registrar_resultado(
                        f"Sistema: acción '{decision.accion}' ejecutada.{detalle} "
                        "Evalúa la nueva captura."
                    )
                else:
                    fallos_seguidos += 1
                    self.stats["fallos"] += 1
                    self.cerebro.registrar_resultado(
                        f"Sistema: la acción '{decision.accion}' falló o no es válida.{detalle} Prueba otra."
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

    # ── Ejecución serializada (consola + entrenador no se pisan) ─────────────────
    def ejecutar_protegido(self, objetivo: str, **kw) -> str:
        """Ejecuta una tarea tomando el lock físico (una sola tarea controla el SO)."""
        with self._exec_lock:
            return self.ejecutar_tarea(objetivo, **kw)

    # ── Modo Entrenador: consume tareas de tasks/active.json ─────────────────────
    def consumir_entrenador(self, stop_event: threading.Event) -> None:
        """
        Bucle del consumidor (hilo aparte). Vigila tasks/active.json: cuando el
        Entrenador deja una tarea en estado 'pendiente', la reclama, la ejecuta y
        la marca 'ejecutada' para que el Entrenador la verifique. Convive con la
        consola manual (ambos comparten el lock físico). Solo se usa con --trainer.
        """
        if not _TRAINER_OK:
            return
        logger.info("[ENTRENADOR] Consumidor activo — esperando tareas en tasks/active.json.")
        while not stop_event.is_set():
            try:
                a = _proto.leer_activa()
                if not a or a.get("estado") != _proto.PENDIENTE:
                    stop_event.wait(0.5)
                    continue

                # Reclamar la tarea (pendiente → activa).
                a["estado"] = _proto.ACTIVA
                a["iniciado_en"] = time.time()
                _proto.escribir_activa(a)
                logger.info("[ENTRENADOR] Tarea recibida: «%s».", a.get("texto", ""))

                resultado = self.ejecutar_protegido(a.get("texto", ""))

                # Marcar ejecutada (el Entrenador la verificará con visión).
                # Si la ranura ya no contiene NUESTRA tarea (expiró y, tras la
                # gracia, el entrenador la liberó y promovió otra), no escribir:
                # se pisaría la tarea nueva.
                actual = _proto.leer_activa()
                if actual is None or actual.get("id") != a.get("id"):
                    logger.warning("[ENTRENADOR] La ranura cambió durante la ejecución "
                                   "(tarea expirada y reasignada) — resultado descartado.")
                else:
                    a["estado"] = _proto.EJECUTADA
                    a["resultado_aria"] = resultado
                    a["acciones_modelo"] = dict(self.ultimas_acciones_modelo)
                    a["modelo_final"] = self.cerebro.ultimo_modelo
                    a["terminado_en"] = time.time()
                    _proto.escribir_activa(a)
                    logger.info("[ENTRENADOR] Tarea ejecutada (%s) [%s | G:%d L:%d O:%d] — esperando verificación.",
                                resultado, self.cerebro.ultimo_modelo,
                                self.ultimas_acciones_modelo.get(MODELO_GEMINI, 0),
                                self.ultimas_acciones_modelo.get(MODELO_NIM_LLAMA, 0),
                                self.ultimas_acciones_modelo.get(MODELO_NIM_OMNI, 0))

                if resultado == LIMITE:
                    logger.warning("[ENTRENADOR] 429: Aria detiene el consumo de tareas.")
                    self._limite_event.set()
                    return
            except Exception as exc:                # noqa: BLE001
                logger.error("[ENTRENADOR] Error en el consumidor: %s", exc)
                stop_event.wait(1.0)

    # ── Util ─────────────────────────────────────────────────────────────────────
    def _dormir(self, segundos: float) -> None:
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

def _banner(aria: Aria, trainer: bool) -> None:
    modo = "  MODO ENTRENADOR activo (consumiendo tasks/active.json)\n" if trainer else ""
    print(
        f"\n{'═' * 64}\n"
        f"  {AGENT_NAME} {AGENT_VERSION} — Agente autónomo de PC (Gemini 2.5 Flash)\n"
        f"  Avatar: {'ACTIVO' if aria.avatar.activo else 'inactivo'} | "
        f"Físico: {'SIMULACIÓN' if aria.controller.simulacion else 'REAL'}\n"
        f"{modo}"
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
        res = aria.ejecutar_protegido(snap.objetivo, ciclo_inicial=snap.ciclo, reanudada=True)
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
    trainer = "--trainer" in sys.argv[1:]
    if trainer and not _TRAINER_OK:
        logger.warning("Se pidió --trainer pero el paquete trainer/ no está disponible; "
                       "se sigue solo en modo manual.")
        trainer = False

    aria = Aria()
    _banner(aria, trainer)
    _resolver_pendiente(aria)

    # Hilo consumidor del Entrenador (solo en modo --trainer).
    stop_event = threading.Event()
    hilo_consumidor = None
    if trainer:
        _proto.inicializar()
        hilo_consumidor = threading.Thread(
            target=aria.consumir_entrenador, args=(stop_event,),
            name="AriaConsumidorEntrenador", daemon=True,
        )
        hilo_consumidor.start()

    try:
        while True:
            try:
                entrada = input("Tarea > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            # En modo entrenador, si Aria tocó 429 en una tarea consumida, salir.
            if aria._limite_event.is_set():
                print("\n  ⏸ Límite de API (429) alcanzado consumiendo tareas. Deteniendo.\n")
                break

            if not entrada:
                continue
            low = entrada.lower()
            if low in ("salir", "exit", "quit", "q"):
                break
            if low in ("estado", "status"):
                print(f"  → {aria.estado_resumen()}")
                continue

            resultado = aria.ejecutar_protegido(entrada)
            _reportar(resultado)

            if resultado == LIMITE:
                print("  Deteniendo por límite de API. Vuelve a ejecutar para reanudar.")
                break
            if resultado == COMPLETADO:
                estado_persistente.limpiar()
    finally:
        stop_event.set()
        if hilo_consumidor is not None:
            hilo_consumidor.join(timeout=2.0)
        aria.detener()
        print(f"\n{AGENT_NAME}: hasta la próxima. ✦\n")


if __name__ == "__main__":
    main()
