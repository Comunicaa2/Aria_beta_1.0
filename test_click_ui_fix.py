"""Check mínimo del fix de click_ui (caso real del log t66b98892) y de
_limpiar_accion. Sin frameworks: python test_click_ui_fix.py → OK o traceback."""

from agent.controller import (_buscar_por_palabra, _es_control_accionable,
                              _CONTROL_TYPES_ACCIONABLES)
from core.brain import _limpiar_accion


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
    print("OK")


if __name__ == "__main__":
    demo()
