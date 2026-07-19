"""[archivos] Lista archivos de una carpeta con tamaño y fecha. Argumentos: RUTA [PATRON] (defecto: workspace, *)."""
import glob
import os
import sys
import time

ruta = os.path.expandvars(os.path.expanduser(sys.argv[1])) if len(sys.argv) > 1 else "."
patron = sys.argv[2] if len(sys.argv) > 2 else "*"
entradas = sorted(glob.glob(os.path.join(ruta, patron)))
if not entradas:
    print(f"Nada que coincida con '{patron}' en {os.path.abspath(ruta)}")
for e in entradas[:40]:
    if os.path.isdir(e):
        print(f"[DIR]  {os.path.basename(e)}")
    else:
        st = os.stat(e)
        fecha = time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))
        print(f"{st.st_size / 1024:8.1f} KB  {fecha}  {os.path.basename(e)}")
if len(entradas) > 40:
    print(f"... y {len(entradas) - 40} mas")
