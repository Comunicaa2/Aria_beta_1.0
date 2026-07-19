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
    <objetivo>  Aria lo persigue HASTA COMPLETARLO ("/goal permanente"): si un
                intento se agota o aborta, aprende la lección y reintenta con otra
                estrategia; si toca el límite de API (429), espera y reanuda.
                Ctrl+C cancela el objetivo en curso (queda como pendiente).
    estado      muestra FSM, avatar y telemetría
    salir       detiene Aria limpiamente
"""

import logging
import sys
import threading
import time

# Consola robusta: el banner usa '═' y emojis; con la salida redirigida (log) o
# en cmd.exe (cp1252) esto crashearía. Mismo fix que trainer/entrenador.py.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                  # noqa: BLE001
    pass

from config import (
    AGENT_NAME,
    AGENT_VERSION,
    AHORRO_IMG_SIZE,
    AHORRO_JPEG_Q,
    BACKOFF_BASE,
    BACKOFF_MAX,
    DELAY_ESTABILIDAD,
    EXTRA_CICLOS,
    LOG_LEVEL,
    MAX_INTENTOS_OBJETIVO,
    MAX_PASOS_TAREA,
    PAUSA_OVERLOADED,
)
from core.brain import (Cerebro, LimiteAPIError,
                        MODELO_GEMINI, MODELO_NIM_LLAMA, MODELO_NIM_OMNI)
from core.fsm import FSM, Estado
from core import memoria
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
CANCELADO  = "cancelado"   # Ctrl+C del usuario durante un objetivo persistente


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
        # Cierre del último intento (motivo del fallo + lección destilada): lo
        # consume la memoria RAG y la coletilla de reintento del objetivo.
        self.ultimo_motivo = ""
        self.ultima_leccion = ""

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
            self.cerebro.reset(objetivo)   # refresca skills/lecciones + memoria RAG
        self.stats["tareas"] += 1
        fallos_seguidos = 0
        # FIX #6: detectar atasco por acción idéntica "exitosa" repetida
        # (p. ej. click en coords erróneas que no cambia nada en pantalla).
        ultima_accion, repes_accion = "", 0
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
                    time.sleep(PAUSA_OVERLOADED)

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
                        self._aprender(objetivo, "respuestas sin formato válido repetidas")
                        return ABORTADO
                    continue

                if decision.pensamiento:
                    logger.info("💭 %s", decision.pensamiento[:120])

                # ── 3. ACTUACIÓN ───────────────────────────────────────────
                self.fsm.a_working()
                # Indicador visible del modelo que decidió esta acción.
                logger.info("[%s] ▶ ACCION: %s", decision.modelo, decision.accion)
                ok = self.controller.ejecutar(decision.accion, cap, decision.contenido)
                self.stats["acciones"] += 1

                if self.controller.es_done:
                    # FIX #3: no aceptar 'done' a ciegas — recapturar y confirmar.
                    cap2 = image.capturar()
                    b64 = cap2.b64 if (cap2 and cap2.b64) else cap.b64
                    if self.cerebro.confirmar_done(objetivo, b64):
                        logger.info("Tarea COMPLETA confirmada (ciclo %d).", ciclo)
                        # Automejora: completada pero LENTA → destilar cómo hacerla
                        # en menos pasos la próxima vez (skill, otro camino, etc.).
                        if ciclo >= 8:
                            self._aprender(
                                objetivo,
                                f"COMPLETADA pero lenta ({ciclo} de {MAX_PASOS_TAREA} "
                                "ciclos); cómo lograrla en menos pasos")
                        self._milla_extra(objetivo)
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
                    # Identidad = acción + contenido: regrabar un script CORREGIDO
                    # con 'guardar' no cuenta como repetición.
                    accion_id = f"{decision.accion}\x00{decision.contenido}"
                    if accion_id == ultima_accion:
                        repes_accion += 1
                    else:
                        ultima_accion, repes_accion = accion_id, 1
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
                    # FIX #6: la 3ª acción idéntica seguida cuenta como fallo aunque
                    # se "ejecute con éxito" — alimenta el circuit-breaker existente.
                    # ponytail: sin comparar capturas; 5 repeticiones legítimas
                    # (p. ej. scroll) abortan — comparar cap.b64 si eso molesta.
                    if repes_accion >= 3:
                        fallos_seguidos += 1
                        self.stats["fallos"] += 1
                        logger.warning("Acción idéntica repetida %d veces sin avance "
                                       "(fallo %d).", repes_accion, fallos_seguidos)
                        self.cerebro.registrar_resultado(
                            f"Sistema: llevas {repes_accion} veces seguidas la MISMA "
                            f"acción '{decision.accion}' y la pantalla no avanza. NO la "
                            "repitas: cambia de estrategia (click_ui \"<nombre del "
                            "control>\" o find_text)."
                        )
                        if fallos_seguidos >= 3:
                            logger.error("Atascada repitiendo la misma acción — abortando tarea.")
                            self._aprender(objetivo, "repitió la misma acción sin efecto")
                            return ABORTADO
                    else:
                        fallos_seguidos = 0
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
                    if fallos_seguidos >= 3:
                        logger.error("Demasiados fallos de ejecución consecutivos — abortando tarea.")
                        self._aprender(objetivo, "acciones fallidas consecutivas")
                        return ABORTADO

            logger.info("Se alcanzó el tope de %d ciclos.", MAX_PASOS_TAREA)
            self._aprender(objetivo, "agotó los ciclos sin completarla")
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

    # ── Milla extra: superar lo pedido (tras un 'done' confirmado) ──────────────
    def _milla_extra(self, objetivo: str) -> None:
        """Fase opcional post-done: hasta EXTRA_CICLOS acciones pequeñas y seguras
        que superen lo pedido (regla 10 del prompt). Jamás convierte el éxito en
        fallo: cualquier error — incluido un 429 — se ignora y la tarea sigue
        COMPLETADA. No toca fallos_seguidos ni el detector de repetición."""
        if EXTRA_CICLOS <= 0:
            return
        try:
            logger.info("Fase EXTRA: buscando mejora opcional (máx %d ciclos).",
                        EXTRA_CICLOS)
            self.cerebro.registrar_resultado(
                "Sistema: objetivo cumplido y CONFIRMADO. Fase EXTRA opcional: si "
                "existe UNA mejora pequeña, segura y relacionada que supere lo "
                "pedido, ejecútala; si no, responde ACCION: done.")
            for i in range(1, EXTRA_CICLOS + 1):
                cap = image.capturar()
                if cap is None or not cap.b64:
                    return
                decision = self.cerebro.pensar(
                    f"Tarea: {objetivo}\nFase EXTRA {i}/{EXTRA_CICLOS} — mejora "
                    f"opcional (la imagen mide {cap.ancho_img}x{cap.alto_img} px). "
                    "¿Mejora pequeña y segura, o done?", cap.b64, profundo=False)
                if not decision.valida or decision.accion.lower().startswith("done"):
                    return
                logger.info("✨ Milla extra (%d/%d): %s", i, EXTRA_CICLOS, decision.accion)
                if not self.controller.ejecutar(decision.accion, cap, decision.contenido):
                    return
                self.stats["acciones"] += 1
                image.esperar_estabilidad(max_seg=DELAY_ESTABILIDAD + 1.5, min_espera=0.5)
                self.cerebro.registrar_resultado(
                    "Sistema: mejora aplicada. ¿Otra mejora pequeña, o done?")
        except LimiteAPIError:
            logger.info("Milla extra interrumpida por 429 — la tarea sigue COMPLETADA.")
        except Exception:                          # noqa: BLE001
            logger.debug("Milla extra falló — se ignora.", exc_info=True)

    # ── Aprendizaje ─────────────────────────────────────────────────────────────
    def _aprender(self, objetivo: str, motivo: str) -> None:
        """Al terminar una tarea fallida (o completada pero lenta), destila una
        lección y la persiste (core/lecciones.py) para inyectarla en el prompt de
        tareas futuras. Best-effort: jamás interfiere con el cierre de la tarea."""
        self.ultimo_motivo = motivo               # lo consumen RAG y reintentos
        try:
            regla = self.cerebro.aprender_leccion(objetivo, motivo)
            if regla:
                logger.info("📚 Lección aprendida: %s", regla)
                self.ultima_leccion = regla
        except Exception:                          # noqa: BLE001
            logger.debug("No se pudo generar lección.", exc_info=True)

    # ── Persistencia ────────────────────────────────────────────────────────────
    def _guardar_pendiente(self, objetivo: str, ciclo: int) -> None:
        snap = estado_persistente.EstadoGuardado(
            objetivo=objetivo,
            ciclo=ciclo,
            completado=False,
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
        """Ejecuta una tarea tomando el lock físico (una sola tarea controla el SO)
        y registra el episodio en la memoria RAG al cerrar (best-effort). Es el
        camino común de consola, objetivo persistente y entrenador: todos dejan
        recuerdo."""
        with self._exec_lock:
            self.ultimo_motivo = self.ultima_leccion = ""
            ciclos_antes = self.stats["ciclos"]
            res = self.ejecutar_tarea(objetivo, **kw)
            memoria.registrar_episodio(
                objetivo, res, motivo=self.ultimo_motivo,
                leccion=self.ultima_leccion,
                ciclos=self.stats["ciclos"] - ciclos_antes)
            return res

    # ── Objetivo persistente ("/goal permanente") ────────────────────────────────
    def perseguir_objetivo(self, objetivo: str, ciclo_inicial: int = 1,
                           reanudada: bool = False) -> str:
        """Bucle exterior del "/goal permanente": no se rinde hasta COMPLETADO.
        AGOTADO/ABORTADO → nuevo intento con el motivo del fallo en el prompt (la
        lección ya quedó destilada y entra fresca vía reset); LIMITE (429) →
        espera con backoff exponencial y reanuda desde el estado guardado.
        Ctrl+C cancela el objetivo (queda como pendiente). MAX_INTENTOS_OBJETIVO=0
        significa sin límite; un valor > 0 acota los intentos (útil en pruebas)."""
        intento, espera, coletilla = 0, BACKOFF_BASE, ""
        try:
            while True:
                intento += 1
                if intento > 1:
                    logger.info("Objetivo «%s» — intento %d.", objetivo[:60], intento)
                res = self.ejecutar_protegido(objetivo + coletilla,
                                              ciclo_inicial=ciclo_inicial,
                                              reanudada=reanudada)
                ciclo_inicial, reanudada = 1, False
                if res == COMPLETADO:
                    estado_persistente.limpiar()
                    return COMPLETADO
                if res == LIMITE:
                    # Cuota, no estrategia: el 429 no cuenta como intento fallido.
                    intento -= 1
                    logger.warning("Límite de API — reanudo el objetivo en %.0fs.", espera)
                    time.sleep(espera)
                    espera = min(espera * 2, BACKOFF_MAX)
                    snap = self.cargar_pendiente()
                    if snap is not None:
                        ciclo_inicial, reanudada = snap.ciclo, True
                    continue
                # AGOTADO / ABORTADO → reintentar con el contexto del fallo.
                if MAX_INTENTOS_OBJETIVO and intento >= MAX_INTENTOS_OBJETIVO:
                    logger.error("Objetivo «%s»: %d intentos sin éxito — me rindo.",
                                 objetivo[:60], intento)
                    estado_persistente.limpiar()
                    return res
                motivo = self.ultimo_motivo or res
                coletilla = (f" (intento {intento + 1}; el anterior terminó en "
                             f"{res} por: {motivo}. Usa otra estrategia)")
                logger.warning("Intento %d terminó en %s — reintento en %.0fs.",
                               intento, res, espera)
                time.sleep(espera)
                espera = min(espera * 2, BACKOFF_MAX)
        except KeyboardInterrupt:
            logger.warning("Objetivo «%s» cancelado por el usuario (Ctrl+C).",
                           objetivo[:60])
            self._guardar_pendiente(objetivo, 1)
            return CANCELADO

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
        res = aria.perseguir_objetivo(snap.objetivo, ciclo_inicial=snap.ciclo,
                                      reanudada=True)
        _reportar(res)
    else:
        estado_persistente.limpiar()
        print("  (Tarea pendiente descartada.)")


def _reportar(resultado: str) -> None:
    msg = {
        COMPLETADO: "✓ Objetivo completado.",
        AGOTADO:    "⚠ Se agotaron los intentos sin completar el objetivo.",
        LIMITE:     "⏸ Límite de API (429): estado guardado. Reinicia para continuar.",
        ABORTADO:   "✗ Objetivo abortado tras agotar los intentos.",
        CANCELADO:  "✋ Objetivo cancelado (Ctrl+C). Quedó guardado como pendiente.",
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

            # "/goal permanente": Aria persigue el objetivo hasta completarlo
            # (reintentos + esperas de cuota dentro); Ctrl+C lo cancela.
            resultado = aria.perseguir_objetivo(entrada)
            _reportar(resultado)
    finally:
        stop_event.set()
        if hilo_consumidor is not None:
            hilo_consumidor.join(timeout=2.0)
        aria.detener()
        print(f"\n{AGENT_NAME}: hasta la próxima. ✦\n")


if __name__ == "__main__":
    main()
