"""[red] Comprueba conectividad y latencia HTTP. Argumentos opcionales: URLs a probar."""
import sys
import time

import httpx

urls = sys.argv[1:] or ["https://www.google.com", "https://api.github.com"]
for u in urls:
    t0 = time.time()
    try:
        r = httpx.head(u, timeout=5, follow_redirects=True)
        print(f"{u} -> HTTP {r.status_code} en {(time.time() - t0) * 1000:.0f} ms")
    except Exception as exc:                       # noqa: BLE001
        print(f"{u} -> SIN CONEXION ({type(exc).__name__})")
