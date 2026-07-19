"""[archivos] Muestra las últimas N líneas de un archivo de texto. Argumentos: RUTA [N] (defecto 30)."""
import os
import sys

if len(sys.argv) < 2:
    print("Uso: skill_leer_archivo.py RUTA [N]")
    sys.exit(1)
ruta = os.path.expandvars(os.path.expanduser(sys.argv[1]))
n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
try:
    with open(ruta, "r", encoding="utf-8", errors="replace") as f:
        lineas = f.readlines()
except OSError as exc:
    print(f"No se pudo leer {ruta}: {exc}")
    sys.exit(1)
print(f"{os.path.basename(ruta)} — {len(lineas)} lineas, ultimas {min(n, len(lineas))}:")
for linea in lineas[-n:]:
    print(linea.rstrip()[:200])
