"""[desktop] Lee el portapapeles; o escribe en él si le pasas texto como argumentos."""
import os
import subprocess
import sys

if len(sys.argv) > 1:
    texto = " ".join(sys.argv[1:])
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", "Set-Clipboard -Value $env:ARIA_CLIP"],
        env={**os.environ, "ARIA_CLIP": texto}, check=False,
    )
    print(f"Copiado al portapapeles ({len(texto)} caracteres).")
else:
    r = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-Clipboard"],
                       capture_output=True, text=True, check=False)
    print((r.stdout or "").strip()[:600] or "(portapapeles vacio)")
