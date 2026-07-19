"""[procesos] Cierra un proceso por nombre exacto (ej: notepad.exe). Argumento: NOMBRE."""
import sys

import psutil

# Procesos críticos del sistema: jamás se tocan.
_CRITICOS = {"system", "system idle process", "csrss.exe", "winlogon.exe",
             "lsass.exe", "svchost.exe", "wininit.exe", "services.exe",
             "smss.exe", "explorer.exe"}

if len(sys.argv) < 2:
    print("Uso: skill_cerrar_proceso.py NOMBRE.exe")
    sys.exit(1)
objetivo = sys.argv[1].lower()
if objetivo in _CRITICOS:
    print(f"RECHAZADO: '{objetivo}' es un proceso critico del sistema.")
    sys.exit(1)
cerrados = 0
for p in psutil.process_iter(["name"]):
    try:
        if (p.info["name"] or "").lower() == objetivo:
            p.terminate()
            cerrados += 1
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
print(f"Procesos '{objetivo}' terminados: {cerrados}"
      if cerrados else f"No hay ningun proceso llamado '{objetivo}'.")
