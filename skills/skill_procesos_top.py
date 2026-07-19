"""[procesos] Top procesos por consumo de RAM. Argumento opcional: N (defecto 8)."""
import sys

import psutil

n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
procs = []
for p in psutil.process_iter(["name", "memory_info"]):
    try:
        procs.append((p.info["memory_info"].rss, p.info["name"] or "?"))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
for rss, nombre in sorted(procs, reverse=True)[:n]:
    print(f"- {nombre}: {rss / 2**20:.0f} MB")
