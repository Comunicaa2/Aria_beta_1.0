"""[archivos] Ordena una carpeta moviendo cada archivo a una subcarpeta según su extensión. Argumento: RUTA."""
import os
import shutil
import sys

if len(sys.argv) < 2:
    print("Uso: skill_organizar_por_tipo.py RUTA")
    sys.exit(1)
ruta = os.path.expandvars(os.path.expanduser(sys.argv[1]))
if not os.path.isdir(ruta):
    print(f"No existe la carpeta {ruta}")
    sys.exit(1)
movidos: dict[str, int] = {}
for nombre in os.listdir(ruta):
    origen = os.path.join(ruta, nombre)
    if not os.path.isfile(origen):
        continue
    ext = os.path.splitext(nombre)[1].lstrip(".").lower() or "sin_extension"
    destino_dir = os.path.join(ruta, ext)
    os.makedirs(destino_dir, exist_ok=True)
    destino = os.path.join(destino_dir, nombre)
    if os.path.exists(destino):                 # jamás sobrescribir
        print(f"OMITIDO (ya existe): {nombre}")
        continue
    shutil.move(origen, destino)
    movidos[ext] = movidos.get(ext, 0) + 1
if movidos:
    for ext, n in sorted(movidos.items()):
        print(f"{ext}/: {n} archivo(s)")
else:
    print("Nada que ordenar (sin archivos sueltos).")
