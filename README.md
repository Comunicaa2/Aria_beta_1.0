# Aria 1.0 — Sistema reactivo que usa el PC mediante visión (Gemini 3.5 Flash)

Aria no es un agente al uso: es un **sistema reactivo siempre en segundo plano**,
a la espera de peticiones. Cuando recibe una, usa el PC como lo haría una persona:
mira la pantalla (visión LLM) y actúa con mouse y teclado hasta completarla; luego
vuelve al reposo (`IDLE`, cero gasto de API).

Reescritura desde cero. La v0.3 usaba un modelo local (Ollama, `qwen3-vl`); la
**1.0** usa **Gemini 3.5 Flash** directo vía Google AI Studio: multimodal nativo
(imagen + comando en un solo flujo), rápido y sin GPU local.

## Filosofía

- **Rapidez y calidad** por encima de todo.
- **Captura de pantalla en RAM** (mss + Pillow), sin tocar disco.
- **Historial mínimo** y **respuestas cortas** del modelo (formato rígido).
- **Degradación elegante**: si falta una dependencia, ese subsistema se desactiva
  solo y Aria sigue funcionando.

## Arquitectura

```
Aria_beta_1.0/
  config.py            configuración + API key de Gemini
  compartido.py        común Aria/entrenador: dotenv, rate limiter, constantes
  main.py              entrada + consola + ciclo cognitivo (orquestador)
  core/
    brain.py           llama a Gemini con imagen + comando; parsea la respuesta
    fsm.py             máquina de estados (IDLE/WORKING/THINKING/OVERLOADED)
    state.py           guardado/carga de estado (parada limpia ante 429)
  agent/
    controller.py      ejecuta los comandos en el SO (pyautogui)
    telemetry.py       CPU / RAM / temperatura (psutil)
  utils/
    image.py           captura de pantalla en RAM + estabilidad de pantalla
  avatar/
    vts.py             avatar VTuber por WebSocket (opcional)
```

## Ciclo cognitivo

`PERCEPCIÓN (captura RAM) → RAZONAMIENTO (Gemini) → ACTUACIÓN (SO) → repetir`

En cada ciclo Aria captura la pantalla, se la envía a Gemini junto con la tarea, y
ejecuta **una** acción. Tras actuar, vuelve a percibir y decide la siguiente.

### Máquina de estados

| Estado       | Significado                                                        |
|--------------|-------------------------------------------------------------------|
| `IDLE`       | Reposo. **Cero gasto de API**.                                    |
| `THINKING`   | Capturando + razonando con Gemini.                               |
| `WORKING`    | Ejecutando la acción física.                                     |
| `OVERLOADED` | CPU/RAM/temperatura saturados → razonamiento superficial + pausa.|

El presupuesto de "pensamiento" de Gemini se activa solo cuando hace falta
(arranque de tarea o atasco) y se apaga en estado saturado para ir al máximo.

## Formato de respuesta del modelo

```
PENSAMIENTO: <máx. 2 líneas>
ACCION: <un solo comando>
FIN
```

### Comandos válidos

| Comando            | Efecto                                          |
|--------------------|-------------------------------------------------|
| `click X Y`        | Clic izquierdo en (X, Y) **del espacio imagen** |
| `double_click X Y` | Doble clic                                      |
| `type TEXTO`       | Escribe el texto                                |
| `key TECLA`        | Pulsa una tecla (`enter`, `esc`, `tab`, …)      |
| `hotkey A+B`       | Combinación (`win+r`, `ctrl+c`, `alt+f4`)       |
| `wait N`           | Espera N segundos                               |
| `done`             | Señala tarea completada *(extensión necesaria)* |

> Las coordenadas que da el modelo están en el espacio de la **imagen reducida**;
> `controller.py` las reescala al espacio real de la pantalla con los metadatos de
> la `Captura`.

## Parada limpia ante límite de API (429)

Cuando Gemini responde **429**, Aria:
1. Termina la acción en curso.
2. Guarda en `aria_state.json`: tarea pendiente, ciclo, historial (sin imágenes)
   y estadísticas.
3. Se detiene limpiamente.

Al reiniciar, detecta la tarea pendiente y ofrece **continuarla** donde se quedó.

## Requisitos mínimos

| Requisito | Mínimo |
|-----------|--------|
| SO | Windows 10/11 (usa UI Automation, `os.startfile` y pywinauto — solo Windows) |
| Python | 3.10 o superior |
| Hardware | Cualquier PC que mueva Windows con soltura (~4 GB RAM libres); **no requiere GPU**, la inferencia es en la nube |
| Red | Conexión a internet estable (cada ciclo sube una captura a Gemini) |
| API key | `GEMINI_API_KEY` de [Google AI Studio](https://aistudio.google.com/) (el tier gratuito sirve) |

Opcionales (se degradan solos si faltan):
- **Tesseract OCR** instalado en el SO — habilita `find_text`.
- **`NVIDIA_API_KEY`** — fallback NIM cuando Gemini devuelve 429 (solo en modo entrenamiento).
- **VTube Studio** con API activada — avatar reactivo.

## Instalación y uso

```bat
pip install -r requirements.txt
python main.py
```

O doble clic en `iniciar.bat`.

### API key

La clave **no** está en el código: se lee de la variable de entorno
`GEMINI_API_KEY`, que se carga desde el archivo `.env` (ignorado por git). Copia
la plantilla y pon tu clave:

```bat
copy .env.example .env
REM edita .env y pon: GEMINI_API_KEY=tu_clave_aqui
python main.py
```

Si `GEMINI_API_KEY` no está definida, Aria falla al arrancar con el mensaje
`Configura GEMINI_API_KEY en el archivo .env`. También puedes exportarla como
variable de entorno real (tiene prioridad sobre el `.env`).

## Avatar VTuber (opcional)

Requiere VTube Studio con la API activada (puerto 8001) y estos hotkeys tipo
*Toggle Expression*: `aria_pensando`, `aria_concentrada`, `aria_panico`.
El avatar refleja el estado de la FSM y se mueve con ondas matemáticas en función
de la telemetría (la cabeza cabecea más cuanto más sube la CPU). Si VTube Studio
no está abierto, el avatar simplemente queda inactivo.
