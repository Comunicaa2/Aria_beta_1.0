"""[ventanas] Lista las ventanas abiertas con posición y estado. Sin argumentos."""
from pywinauto import Desktop

for w in Desktop(backend="uia").windows():
    # Cada ventana se procesa de forma independiente: algunas exponen interfaces
    # COM rotas (NULL pointer) y no deben tumbar el listado completo.
    try:
        t = (w.window_text() or "").strip()
        if not t:
            continue
        r = w.rectangle()
        extra = " (minimizada)" if w.is_minimized() else ""
        print(f"- {t[:60]} [{r.left},{r.top} {r.width()}x{r.height()}]{extra}")
    except Exception:                              # noqa: BLE001
        continue
