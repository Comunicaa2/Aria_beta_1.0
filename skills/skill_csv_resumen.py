"""[datos] Resumen estadístico de un CSV: filas, columnas y media/min/max de las numéricas. Argumento: RUTA."""
import csv
import statistics
import sys

if len(sys.argv) < 2:
    print("Uso: skill_csv_resumen.py RUTA.csv")
    sys.exit(1)
with open(sys.argv[1], "r", encoding="utf-8-sig", errors="replace", newline="") as f:
    filas = list(csv.DictReader(f))
if not filas:
    print("CSV vacio o sin cabecera.")
    sys.exit(1)
print(f"Filas: {len(filas)} | Columnas: {', '.join(filas[0].keys())}")
for col in list(filas[0].keys())[:8]:
    valores = []
    for fila in filas:
        try:
            valores.append(float((fila[col] or "").replace(",", ".")))
        except ValueError:
            pass
    if len(valores) >= len(filas) * 0.5:          # columna mayormente numérica
        print(f"{col}: media={statistics.fmean(valores):.4g} "
              f"min={min(valores):.4g} max={max(valores):.4g}")
