"""[archivos] Busca archivos por patrón de forma recursiva. Argumentos: RAIZ PATRON (ej: C:\\Users\\yo *.csv)."""
import fnmatch
import os
import sys

if len(sys.argv) < 3:
    print("Uso: skill_buscar_archivos.py RAIZ PATRON")
    sys.exit(1)
raiz = os.path.expandvars(os.path.expanduser(sys.argv[1]))
patron = sys.argv[2]
tope = 30
hallados = 0
for base, _dirs, archivos in os.walk(raiz):
    for a in archivos:
        if fnmatch.fnmatch(a.lower(), patron.lower()):
            print(os.path.join(base, a))
            hallados += 1
            if hallados >= tope:
                print(f"... (tope de {tope} resultados)")
                sys.exit(0)
print(f"Total: {hallados}" if hallados else f"Sin coincidencias de '{patron}' bajo {raiz}")
