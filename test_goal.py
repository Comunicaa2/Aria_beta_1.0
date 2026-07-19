"""Check mínimo del objetivo persistente ("/goal permanente"), la milla extra y
la memoria RAG. Sin red y sin tocar el estado real: stubs planos + asserts.

Uso: python test_goal.py
"""

import json
import os
import tempfile
import types

import main
from core import memoria


def _aria_stub() -> "main.Aria":
    """Aria sin __init__ (nada de avatar/telemetría/httpx): solo lo que usan
    perseguir_objetivo y _milla_extra."""
    a = object.__new__(main.Aria)
    a.stats = {"tareas": 0, "ciclos": 0, "acciones": 0, "fallos": 0, "limite_api": 0}
    a.ultimo_motivo = ""
    a.ultima_leccion = ""
    return a


class _Parches:
    """Sustituye time.sleep y estado_persistente.limpiar de main; restaura al salir."""

    def __enter__(self):
        self.esperas: list[float] = []
        self._sleep = main.time.sleep
        self._limpiar = main.estado_persistente.limpiar
        main.time.sleep = self.esperas.append
        main.estado_persistente.limpiar = lambda: None
        return self

    def __exit__(self, *exc):
        main.time.sleep = self._sleep
        main.estado_persistente.limpiar = self._limpiar


def test_goal_reintenta_hasta_completar():
    a = _aria_stub()
    guion = [main.ABORTADO, main.ABORTADO, main.COMPLETADO]
    llamadas: list[tuple] = []

    def fake_exec(obj, **kw):
        llamadas.append((obj, kw))
        a.ultimo_motivo = "repitió la misma acción"
        return guion.pop(0)

    a.ejecutar_protegido = fake_exec
    with _Parches() as p:
        res = a.perseguir_objetivo("abre la calculadora")
    assert res == main.COMPLETADO
    assert len(llamadas) == 3, f"esperaba 3 intentos, hubo {len(llamadas)}"
    # La coletilla lleva el número de intento y el motivo del fallo anterior.
    assert "abre la calculadora" in llamadas[1][0]
    assert "intento 2" in llamadas[1][0] and "repitió la misma acción" in llamadas[1][0]
    assert "intento 3" in llamadas[2][0]
    # Backoff exponencial entre reintentos.
    assert p.esperas == [main.BACKOFF_BASE, min(main.BACKOFF_BASE * 2, main.BACKOFF_MAX)]
    print("OK goal: 2 fallos + reintento con coletilla y backoff → COMPLETADO")


def test_goal_reanuda_tras_429():
    a = _aria_stub()
    guion = [main.LIMITE, main.COMPLETADO]
    llamadas: list[dict] = []
    a.ejecutar_protegido = lambda obj, **kw: (llamadas.append(kw), guion.pop(0))[1]
    a.cargar_pendiente = lambda: types.SimpleNamespace(ciclo=7)
    with _Parches() as p:
        res = a.perseguir_objetivo("tarea larga")
    assert res == main.COMPLETADO
    assert llamadas[1] == {"ciclo_inicial": 7, "reanudada": True}, llamadas[1]
    assert p.esperas == [main.BACKOFF_BASE]
    print("OK goal: 429 → espera y reanuda desde el ciclo 7 del snapshot")


def test_goal_respeta_max_intentos():
    a = _aria_stub()
    a.ejecutar_protegido = lambda obj, **kw: main.ABORTADO
    viejo = main.MAX_INTENTOS_OBJETIVO
    main.MAX_INTENTOS_OBJETIVO = 2
    try:
        with _Parches():
            res = a.perseguir_objetivo("imposible")
    finally:
        main.MAX_INTENTOS_OBJETIVO = viejo
    assert res == main.ABORTADO
    print("OK goal: MAX_INTENTOS acota los reintentos")


def test_milla_extra_no_rompe_el_exito():
    a = _aria_stub()

    class CerebroRoto:
        def registrar_resultado(self, nota):
            raise RuntimeError("boom")

    a.cerebro = CerebroRoto()
    # No debe lanzar: el extra jamás convierte un éxito en fallo.
    assert a._milla_extra("lo que sea") is None
    print("OK milla extra: un fallo interno se ignora (el éxito se conserva)")


def test_memoria_rag():
    ruta_real, embed_real = memoria.RUTA, memoria._embed
    fd, tmp = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    memoria.RUTA = tmp
    try:
        # Sin red (embed → None): registra y recupera por solape de palabras.
        memoria._embed = lambda t: None
        memoria.registrar_episodio("abrir el bloc de notas y escribir", "abortado",
                                   motivo="repitió la misma acción",
                                   leccion="usa click_ui por nombre", ciclos=12)
        memoria.registrar_episodio("calcular una suma con la calculadora", "completado")
        memoria.registrar_episodio("da igual", "limite")   # los 429 NO se guardan
        eps = memoria.cargar()
        assert len(eps) == 2, f"esperaba 2 episodios, hay {len(eps)}"

        rel = memoria.relevantes("escribir una nota en el bloc de notas", k=2)
        assert rel and "bloc" in rel[0]["objetivo"], rel
        assert all("calculadora" not in e["objetivo"] for e in rel)

        # Con vectores fijos manda el coseno.
        eps[0]["vec"], eps[1]["vec"] = [1.0, 0.0], [0.0, 1.0]
        with open(tmp, "w", encoding="utf-8") as f:
            for e in eps:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        memoria._embed = lambda t: [1.0, 0.0]
        rel = memoria.relevantes("cualquier objetivo", k=1)
        assert rel and rel[0]["objetivo"].startswith("abrir el bloc"), rel

        # La sección de prompt incluye motivo y lección del episodio relevante.
        memoria._embed = lambda t: None
        secc = memoria.seccion_prompt("escribir en el bloc de notas")
        assert "EXPERIENCIA RELEVANTE" in secc
        assert "repitió la misma acción" in secc and "usa click_ui" in secc
        print("OK memoria RAG: registro, top-k por coseno, fallback por palabras y sección")
    finally:
        memoria.RUTA, memoria._embed = ruta_real, embed_real
        os.unlink(tmp)


if __name__ == "__main__":
    test_goal_reintenta_hasta_completar()
    test_goal_reanuda_tras_429()
    test_goal_respeta_max_intentos()
    test_milla_extra_no_rompe_el_exito()
    test_memoria_rag()
    print("\nTodos los checks de test_goal.py pasan.")
