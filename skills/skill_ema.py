"""[datos] Calcula dos EMAs sobre una columna de un CSV (velas/cierres) y detecta su cruce. Argumentos: RUTA [COLUMNA=close] [P_RAPIDA=20] [P_LENTA=50]."""
import csv
import sys

if len(sys.argv) < 2:
    print("Uso: skill_ema.py RUTA.csv [COLUMNA] [P_RAPIDA] [P_LENTA]")
    sys.exit(1)
ruta = sys.argv[1]
columna = sys.argv[2] if len(sys.argv) > 2 else "close"
p1 = int(sys.argv[3]) if len(sys.argv) > 3 else 20
p2 = int(sys.argv[4]) if len(sys.argv) > 4 else 50

with open(ruta, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
    lector = csv.DictReader(f)
    campos = lector.fieldnames or []
    col = next((c for c in campos if c.lower() == columna.lower()), None)
    if col is None:
        print(f"No existe la columna '{columna}'. Columnas: {', '.join(campos)}")
        sys.exit(1)
    precios = []
    for fila in lector:
        try:
            precios.append(float((fila[col] or "").replace(",", ".")))
        except ValueError:
            pass

if len(precios) < p2:
    print(f"Datos insuficientes: {len(precios)} valores, se necesitan >= {p2}.")
    sys.exit(1)


def ema(datos: list, periodo: int) -> list:
    k = 2 / (periodo + 1)
    serie = [sum(datos[:periodo]) / periodo]      # arranque: SMA del periodo
    for x in datos[periodo:]:
        serie.append(x * k + serie[-1] * (1 - k))
    return serie


rapida, lenta = ema(precios, p1), ema(precios, p2)
# Alinear al mismo tramo final para comparar.
n = min(len(rapida), len(lenta))
rapida, lenta = rapida[-n:], lenta[-n:]
print(f"{len(precios)} valores de '{col}' | ultimo precio: {precios[-1]:.6g}")
print(f"EMA{p1}: {rapida[-1]:.6g} | EMA{p2}: {lenta[-1]:.6g}")
estado = "EMA rapida POR ENCIMA de la lenta" if rapida[-1] > lenta[-1] \
         else "EMA rapida POR DEBAJO de la lenta"
cruce = next((i for i in range(n - 1, 0, -1)
              if (rapida[i] > lenta[i]) != (rapida[i - 1] > lenta[i - 1])), None)
print(estado + (f" | ultimo cruce hace {n - 1 - cruce} velas" if cruce else " | sin cruces en el tramo"))
# Solo describe datos: no emite señales de compra/venta ni consejo financiero.
