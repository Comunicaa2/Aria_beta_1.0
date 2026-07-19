"""[sistema] Resumen del sistema: CPU, RAM, discos y uptime. Sin argumentos."""
import time

import psutil

print(f"CPU: {psutil.cpu_percent(interval=0.4):.0f}% ({psutil.cpu_count()} nucleos)")
m = psutil.virtual_memory()
print(f"RAM: {m.percent:.0f}% de {m.total / 2**30:.1f} GB")
for d in psutil.disk_partitions():
    try:
        u = psutil.disk_usage(d.mountpoint)
        print(f"Disco {d.device} {u.percent:.0f}% usado de {u.total / 2**30:.0f} GB")
    except OSError:
        pass
print(f"Encendido hace {(time.time() - psutil.boot_time()) / 3600:.1f} h")
