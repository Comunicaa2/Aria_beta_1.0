"""Check mínimo del fix de click_ui (caso real del log t66b98892), de
_limpiar_accion y de las acciones guardar/ejecutar_python (workspace).
Sin frameworks: python test_click_ui_fix.py → OK o traceback."""

from agent.controller import (_buscar_por_palabra, _es_control_accionable,
                              _CONTROL_TYPES_ACCIONABLES, Controller)
from core.brain import _limpiar_accion, parsear


class _Info:
    def __init__(self, ct): self.control_type = ct


class _Elem:
    def __init__(self, nombre, ct="Text"):
        self._n = nombre
        self.element_info = _Info(ct)

    def window_text(self): return self._n


def demo():
    # ── caso del log: "no" NO debe matchear "Novedades" ni un párrafo largo ──
    boton_no = _Elem("No", "Button")
    arbol = [
        ("Novedades", _Elem("Novedades", "MenuItem")),
        ("¿Quieres guardar los cambios? Si no guardas, se perderán.",
         _Elem("¿Quieres guardar los cambios? Si no guardas, se perderán.")),
        ("Guardar", _Elem("Guardar", "Button")),
        ("No", boton_no),
        ("No guardar nunca", _Elem("No guardar nunca", "Hyperlink")),
    ]
    accionables = [(n, e) for n, e in arbol if _es_control_accionable(e)]
    assert boton_no in [e for _, e in accionables]
    assert not _es_control_accionable(arbol[1][1])          # párrafo Text: fuera

    elem, n = _buscar_por_palabra("no", accionables)
    assert elem is boton_no, f"'no' matcheó {elem.window_text() if elem else None}"
    assert n >= 1

    # "Novedades" no contiene la PALABRA "no"; sin match → (None, 0)
    elem, n = _buscar_por_palabra("xyz", accionables)
    assert (elem, n) == (None, 0)

    # empate genuino → no adivinar
    empate = [("Ok", _Elem("Ok", "Button")), ("OK", _Elem("OK", "Button"))]
    elem, n = _buscar_por_palabra("ok", empate)
    assert elem is None and n == 2

    # ── _limpiar_accion: paridad de comillas ──
    assert _limpiar_accion('click_ui "Cerrar"') == 'click_ui "Cerrar"'
    assert _limpiar_accion("done'}]") == "done"
    assert _limpiar_accion('"done"') == "done"

    assert "Text" not in _CONTROL_TYPES_ACCIONABLES

    # ── bloque CONTENIDO: parseo y cercas de código ──
    d = parsear('PENSAMIENTO: calculo la media\n'
                'ACCION: guardar analisis.py\n'
                'CONTENIDO:\n'
                '```python\n'
                'datos = [1, 2, 3]\n'
                'print(sum(datos) / len(datos))\n'
                '```\n'
                'FIN')
    assert d.accion == "guardar analisis.py"
    assert d.contenido == "datos = [1, 2, 3]\nprint(sum(datos) / len(datos))"
    # sin 'guardar' no se arrastra contenido
    assert parsear("PENSAMIENTO: x\nACCION: click 1 2\nFIN").contenido == ""

    # ── guardar + ejecutar_python: ida y vuelta real en workspace ──
    ctrl = Controller()
    assert ctrl.ejecutar("guardar _test_aria.py",
                         contenido="print(2 + 2)") is True
    assert ctrl.ejecutar("ejecutar_python _test_aria.py") is True
    assert "4" in ctrl.ultimo_detalle
    # script con error → False y el error queda en ultimo_detalle
    ctrl.ejecutar("guardar _test_aria.py", contenido="1/0")
    assert ctrl.ejecutar("ejecutar_python _test_aria.py") is False
    assert "ZeroDivisionError" in ctrl.ultimo_detalle
    # frontera de confianza: rutas y traversal rechazados
    assert ctrl.ejecutar("guardar ..\\fuera.py", contenido="x") is False
    assert ctrl.ejecutar("guardar C:\\Windows\\x.py", contenido="x") is False
    assert ctrl.ejecutar("ejecutar_python ..\\fuera.py") is False
    # guardar sin CONTENIDO → fallo limpio
    assert ctrl.ejecutar("guardar vacio.txt") is False

    # ── skills: catálogo + ejecutar_python con argumentos ──
    from core import skills
    assert ctrl.ejecutar(
        "guardar skill_test.py",
        contenido='"""Suma los numeros que reciba como argumentos."""\n'
                  'import sys\nprint(sum(int(x) for x in sys.argv[1:]))') is True
    catalogo = dict(skills.listar())
    assert "skill_test.py" in catalogo
    assert "Suma los numeros" in catalogo["skill_test.py"]
    assert "skill_test.py" in skills.seccion_prompt()
    assert ctrl.ejecutar("ejecutar_python skill_test.py 2 3 5") is True
    assert "10" in ctrl.ultimo_detalle

    import os
    from config import WORKSPACE_DIR
    os.remove(os.path.join(WORKSPACE_DIR, "_test_aria.py"))
    os.remove(os.path.join(WORKSPACE_DIR, "skill_test.py"))
    print("OK")


if __name__ == "__main__":
    demo()
