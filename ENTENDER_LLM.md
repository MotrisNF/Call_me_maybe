# Entender un LLM desde cero (para *call me maybe*)

Guía de estudio **sin código**. El objetivo es que, cuando termines de leerla y de
hacer los ejercicios, entiendas de verdad qué pasa por dentro de un modelo de
lenguaje y por qué el proyecto se resuelve con **decodificación restringida**
(*constrained decoding*).

No hay que programar nada aquí. Solo leer, pensar, hacer los ejercicios con
lápiz y papel (o en un chat con un LLM) y comprobar tus respuestas.

---

## Parte 0 — Dónde descargar el modelo (y no llenar el disco)

### El problema

Tu carpeta personal es diminuta:

| Sitio | Tamaño total | Libre | Notas |
|---|---|---|---|
| `/home/saperez-` (tu `~`, tu escritorio) | ~4,7 GB | ~2,3 GB | **No cabe el modelo con holgura** |
| `/goinfre/saperez-` (= `~/goinfre`) | ~137 GB | ~100 GB | Disco local, rápido. **Se borra cada cierto tiempo** (reinicios / limpiezas de 42) |
| `/sgoinfre/saperez-` | varios TB | mucho | Más permanente, pero es de red → cargar el modelo es más lento |

El modelo **Qwen/Qwen3-0.6B** ocupa alrededor de **1,5–2 GB** en disco una vez
descargado (pesos del modelo + tokenizer + vocabulario). Si lo descargas en `~`
te quedas sin espacio casi seguro.

**Recomendación:** descárgalo en `~/goinfre`. Si te molesta volver a bajarlo
después de cada limpieza, usa `~/sgoinfre`.

### Cómo funciona la descarga

La clase `Small_LLM_Model` del `llm_sdk` usa por debajo las librerías de
Hugging Face (`transformers` y `huggingface_hub`). Esas librerías **no descargan
al azar**: guardan todo en una carpeta caché. Por defecto esa carpeta es:

```
~/.cache/huggingface
```

…que está dentro de tu `~` pequeño. Para cambiarla solo hay que definir una
**variable de entorno** llamada `HF_HOME` antes de ejecutar nada. Todas las
descargas (pesos, `vocab.json`, `merges.txt`, `tokenizer.json`…) irán a donde tú
digas.

### Pasos concretos (esto es configuración de terminal, no código del proyecto)

1. **Crea la carpeta destino:**

   ```
   mkdir -p ~/Sgoinfre/hf_cache
   ```

2. **Apunta Hugging Face a esa carpeta.** Para que valga siempre, añade esta
   línea al final de tu `~/.hellishrc` (tu shell es `hellish`) — o a `~/.bashrc`:

   ```
   export HF_HOME=~/Sgoinfre/hf_cache
   ```

   Después abre una terminal nueva, o recarga la config, y comprueba:

   ```
   echo $HF_HOME
   ```

   Debe imprimir la ruta de `goinfre`, no algo vacío.

3. **Prepara el entorno del proyecto con `uv`** (ver Parte 8 para qué es `uv`):

   ```
   uv sync
   ```

4. **Descarga el modelo una vez, a mano**, sin ejecutar tu proyecto todavía.
   `huggingface_hub` trae una herramienta de línea de comandos:

   ```
   uv run hf download Qwen/Qwen3-0.6B
   ```

   (En versiones antiguas el comando es `uv run huggingface-cli download Qwen/Qwen3-0.6B`.)

5. **Verifica que se descargó donde querías:**

   ```
   du -sh ~/goinfre/hf_cache
   ls ~/goinfre/hf_cache/hub
   ```

   Deberías ver una carpeta tipo `models--Qwen--Qwen3-0.6B` y que ocupa >1 GB.

6. A partir de aquí, cuando ejecutes el proyecto (`uv run python -m src ...`),
   el SDK **encontrará el modelo ya descargado** en esa caché y no volverá a
   bajarlo, siempre que `HF_HOME` siga apuntando ahí en la terminal desde la que
   lo lanzas.

> Alternativa sin variable de entorno: crear un enlace simbólico
> `ln -s ~/goinfre/hf_cache ~/.cache/huggingface` (borrando antes el
> `~/.cache/huggingface` que exista). Funciona, pero la variable `HF_HOME` es más
> limpia y explícita.

> Si usas `~/goinfre` y un día el modelo "desaparece": no es un bug, es que
> limpiaron `goinfre`. Vuelve a hacer el paso 4.

---

## Parte 1 — Qué es realmente un LLM

Un modelo de lenguaje grande (LLM) hace **una sola cosa**:

> Dada una secuencia de texto, predice **qué viene después**, un trocito cada vez.

No "entiende" en el sentido humano, no consulta una base de datos, no ejecuta
funciones. Es una función matemática gigante que recibe una lista de números y
devuelve, para cada posible "siguiente trocito", una puntuación de
"cómo de probable es que sea este".

Todo lo demás (chatear, resumir, llamar funciones) se construye **encima** de esa
capacidad de predecir el siguiente trocito.

El "pipeline" que describe el subject es exactamente esto:

```
Prompt → Tokenización → Input IDs → LLM → Logits → Elegir el siguiente token → (repetir)
```

Las siguientes partes recorren cada flecha de ese diagrama.

---

## Parte 2 — Tokens: el texto se parte en trozos

El modelo no ve letras ni palabras completas. Ve **tokens**: trozos de palabra.
Un token puede ser una palabra entera (`" cat"`), un trozo (`"Program"`, `"ming"`),
un signo de puntuación (`"?"`), un espacio + palabra, etc.

El proceso de partir texto en tokens se llama **tokenización**. Se hace con un
algoritmo (Qwen usa uno de la familia *BPE*, *byte-pair encoding*) que aprendió,
durante el entrenamiento, qué trozos son frecuentes y merecen ser un token.

Detalles importantes que verás en la práctica:

- Los **espacios se guardan dentro del token**. Muchos tokenizers representan un
  espacio inicial con un símbolo especial (a menudo se ve como `Ġ` o `▁`). Así el
  modelo puede reconstruir el texto exacto, con sus espacios, al de-tokenizar.
  Ejemplo del subject: `"What is the sum"` → `["What", "Ġis", "Ġthe", "Ġsum", ...]`.
- Un mismo texto siempre produce los mismos tokens (con el mismo tokenizer).
- Palabras raras se parten en varios tokens; palabras comunes suelen ser uno.
- Números grandes casi nunca son un token: `345` puede ser `["3", "45"]` o
  `["34", "5"]`, etc. **Esto es clave para el proyecto** (ver Parte 9).

### Ejercicio 2.1

Coge estas frases del fichero de pruebas del proyecto y **predice a mano** en
cuántos trozos crees que se parte cada una y dónde estarían los cortes:

1. `Greet shrek`
2. `What is the square root of 16?`
3. `Replace all vowels in 'Programming is fun' with asterisks`

Luego comprueba tu intuición con un tokenizer online (busca "tiktokenizer" o
"huggingface tokenizer playground" y selecciona un modelo tipo Qwen o GPT-2).
Fíjate especialmente en: ¿dónde van los espacios? ¿`Programming` es un token o
varios? ¿`16` es un token o dos?

### Ejercicio 2.2

En el playground, escribe `16` y luego `160000`. Anota en cuántos tokens se parte
cada uno. Repite con `2`, `20`, `200`, `2000`. **Conclusión que debes sacar:**
el modelo no "ve" el número, ve una secuencia de trozos de dígitos que tiene que
ir generando uno a uno.

---

## Parte 3 — Input IDs: cada token es un número

El modelo tampoco trabaja con los tokens como texto. Cada token tiene un **número
entero** asignado: su *ID*. La lista completa "token ↔ ID" es el **vocabulario**
del modelo, y vive en un fichero (`vocab.json` para BPE, más `merges.txt` con las
reglas de fusión).

- `encode(texto)` = tokenizar **y** convertir a IDs → lista de enteros.
- `decode(lista_de_IDs)` = lo contrario: de números a texto.
- El vocabulario de Qwen3-0.6B tiene ~150.000 entradas. Es decir, en cada paso el
  modelo elige "el siguiente token" entre ~150.000 opciones.

En el SDK esto se corresponde con:

| Método del SDK | Qué te da |
|---|---|
| `encode(text)` | texto → tensor de input IDs |
| `decode(ids)` | input IDs → texto |
| `get_path_to_vocab_file()` | ruta al `vocab.json` (mapa token↔ID) |
| `get_path_to_merges_file()` | ruta al `merges.txt` (reglas BPE) |
| `get_logits_from_input_ids(ids)` | input IDs → logits del siguiente token |

### Ejercicio 3.1

Abre el `vocab.json` del modelo (después de la Parte 0 estará en
`~/goinfre/hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/.../vocab.json`).
Es un JSON gigante `{"token": id, ...}`. Busca (con el buscador de tu editor):

1. El ID del token para `{` (llave de apertura).
2. El ID del token para `"` (comilla doble).
3. El ID del token para `}`.
4. Los IDs de los dígitos `0`, `1`, `2` … `9` sueltos.
5. ¿Existe un token exactamente igual a `":` (comilla + dos puntos)? ¿Y ` "`
   (espacio + comilla)?

Apunta esos IDs en un papel. **Los vas a necesitar de verdad** para la
decodificación restringida: son los "trozos legales" cuando estás construyendo un
JSON.

### Ejercicio 3.2

Piensa (sin código): si `decode([a, b, c])` te devuelve `{"a"`, ¿qué información
te da eso sobre qué IDs son "válidos" como cuarto elemento de la lista si quieres
que el resultado siga siendo un JSON válido? Haz una lista de 4–5 continuaciones
que **sí** valdrían y 4–5 que **no**.

---

## Parte 4 — El modelo: una función que devuelve logits

Metes la lista de input IDs en el modelo. El modelo hace un montón de
multiplicaciones de matrices (la "red neuronal") y saca **un número por cada
token del vocabulario**. Esos ~150.000 números se llaman **logits**.

- Un logit es una **puntuación bruta**, sin normalizar. No es una probabilidad.
- Cuanto **más alto** el logit de un token, más "cree" el modelo que ese token es
  el siguiente.
- Para convertirlos en probabilidades reales (que sumen 1) se aplica una función
  llamada **softmax**. Para el proyecto casi no necesitas softmax: te basta con
  saber que "logit más alto = más probable".

`get_logits_from_input_ids(ids)` te devuelve exactamente esa lista de logits para
**el siguiente** token, dado lo que llevas hasta ahora.

### Ejercicio 4.1

Imagina que, en un paso de generación, el modelo devuelve estos logits (solo te
enseño 6 de los 150.000, el resto son más bajos):

| Token | Logit |
|---|---|
| `}` | 8.1 |
| ` ` (espacio) | 7.9 |
| `,` | 2.3 |
| `"` | 1.0 |
| `x` | -4.2 |
| `\n` | -6.0 |

1. Si eliges **siempre el logit más alto** (esto se llama *greedy* o *argmax*),
   ¿qué token sale?
2. El segundo candidato está muy cerca del primero. ¿Qué dice eso sobre la
   confianza del modelo en este paso?
3. Si tu formato de salida **prohíbe** cerrar la llave ahora mismo (porque aún
   falta un valor), ¿cuál sería el token elegido si tachas los prohibidos?

El punto 3 es, en una frase, **toda la idea del proyecto**.

---

## Parte 5 — El bucle de generación

Un LLM genera **un token cada vez**. El ciclo completo es:

1. Tienes una secuencia de input IDs (tu prompt convertido a números).
2. Se la pasas al modelo → obtienes logits.
3. Eliges un token (por ahora, di que eliges el de logit más alto).
4. **Añades ese token al final** de la secuencia.
5. Vuelves al paso 2 con la secuencia ya más larga.
6. Paras cuando el modelo genera un token especial de "fin" (EOS) o cuando llegas
   a un límite que tú pongas.

El resultado es la concatenación de todos los tokens que fuiste eligiendo.

### Ejercicio 5.1

Traza a mano este bucle. Prompt inicial (ya tokenizado, invento los IDs):
`[10, 11, 12]`. En cada paso te digo qué token elige el modelo:

| Paso | Secuencia de entrada | Token elegido (texto / ID) |
|---|---|---|
| 1 | `[10, 11, 12]` | `{` / 90 |
| 2 | ? | `"` / 91 |
| 3 | ? | `a` / 92 |
| 4 | ? | `"` / 91 |
| 5 | ? | `:` / 93 |
| 6 | ? | EOS |

Rellena la columna "Secuencia de entrada" de cada paso. Escribe el texto final
generado. ¿Es un JSON válido? ¿Qué le falta?

### Ejercicio 5.2

Repite el ejercicio anterior pero ahora **tú** decides los tokens, con esta
regla: el texto generado tiene que acabar siendo exactamente
`{"a": 5}`. Escribe la lista de tokens que elegirías, paso a paso, suponiendo
que existen los tokens `{`, `"`, `a`, `"`, `:`, ` ` (espacio), `5`, `}`.
Fíjate en que has tenido que **tomar una decisión de formato en cada paso**.

---

## Parte 6 — El prompt no es solo tu pregunta

Cuando "hablas" con un LLM de chat, tu texto se envuelve en una **plantilla**
antes de tokenizar: marcas de rol (`system`, `user`, `assistant`), instrucciones,
separadores especiales. El modelo aprendió durante el entrenamiento a responder
después de ese formato concreto.

Para el proyecto, tú controlas el prompt entero. Un buen prompt para este
problema suele incluir:

- Las **definiciones de funciones** disponibles (nombre, parámetros, tipos,
  descripción), tomadas de `functions_definition.json`.
- La **pregunta del usuario** (`prompt` del `function_calling_tests.json`).
- Una instrucción clara de que responda **solo** con un objeto que diga qué
  función llamar y con qué argumentos.
- Puede ayudar mostrarle el formato exacto que esperas (un ejemplo).

### Ejercicio 6.1

Escribe (en texto, a mano) el prompt que le pasarías al modelo para el caso:

> `functions_definition.json` contiene `fn_add_numbers(a: number, b: number)` y
> `fn_greet(name: string)`.
> Pregunta del usuario: `"What is the sum of 2 and 3?"`

Redáctalo entero como si se lo fueras a pegar a un chatbot. Luego pruébalo de
verdad en cualquier chat LLM (aunque sea uno grande) y mira si responde algo
parseable.

### Ejercicio 6.2

Coge tu prompt del 6.1 y cámbialo tres veces:

1. Quita el ejemplo de formato.
2. Pon las funciones **después** de la pregunta en vez de antes.
3. Pídele que "explique su razonamiento antes de responder".

Prueba las tres versiones. Anota cuál da salidas más limpias y más fáciles de
parsear. **Conclusión:** el formato y el orden del prompt cambian mucho el
resultado, pero **nunca te garantizan** un JSON perfecto. Eso lleva a la Parte 7.

---

## Parte 7 — Por qué un modelo pequeño falla al generar JSON

El subject lo dice: un modelo de 0,6 B de parámetros, pidiéndole JSON "por las
buenas", acierta un formato válido quizá el **30 %** de las veces. Fallos típicos:

- Se olvida una comilla o una llave.
- Añade texto antes o después (`"Sure! Here is the JSON: ..."`).
- Inventa un nombre de función que no existe.
- Pone `"a": "2"` (string) cuando el esquema pide `number`.
- Mete comas de más (*trailing comma*), o comentarios.
- Se enrolla y nunca cierra el objeto.

La causa de fondo: en cada paso el modelo **solo** mira "qué token es más
probable según lo que aprendió", y a veces lo más probable rompe el formato.
No hay nadie comprobando la estructura.

### Ejercicio 7.1

En un chat con un modelo **pequeño** (si puedes, uno tipo 0,5–2 B; si no, súbele
la "temperatura" a uno normal para que sea más caótico), pídele **10 veces** la
misma tarea de function calling con el mismo prompt. Cuenta:

- ¿Cuántas respuestas son JSON perfectamente parseable?
- ¿Cuántas eligen la función correcta?
- ¿Cuántas aciertan los tipos de los argumentos?

Vas a ver "a ojo" por qué hace falta la Parte 8.

---

## Parte 8 — Decodificación restringida (el corazón del proyecto)

Idea central:

> No dejes que el modelo elija **cualquier** token. En cada paso, calcula **qué
> tokens mantienen la salida válida** (JSON correcto + que encaje con el esquema),
> y **prohíbe todos los demás** antes de elegir.

"Prohibir" un token = poner su logit a **menos infinito**. Así, al elegir el de
logit más alto (o al muestrear), es **imposible** que salga un token ilegal.

El bucle de la Parte 5, corregido:

1. Secuencia actual de input IDs.
2. Modelo → logits (uno por token del vocabulario).
3. **Nuevo paso:** mira lo que llevas generado. Según las reglas de tu formato,
   decide qué tokens serían válidos como siguiente.
4. A todos los tokens **no** válidos, ponles logit = −∞.
5. Ahora elige (argmax o muestreo) — solo puede salir un token válido.
6. Añádelo a la secuencia. Vuelve al 2.
7. Paras cuando tu formato está "completo" (el JSON está cerrado y cumple el
   esquema).

Resultado: JSON válido el **100 %** de las veces, y además que **cumple el
esquema** (nombre de función real, argumentos correctos, tipos correctos).

### Cómo saber "qué es válido ahora": una máquina de estados

Piensa en construir el JSON como recorrer un camino con reglas. Para la salida
del proyecto, cada objeto es:

```
{ "name": "<uno de los nombres de función>", "parameters": { <args según esa función> } }
```

En cada punto del camino sabes exactamente qué caracteres pueden venir:

- Al principio: solo `{`.
- Después de `{`: solo `"` (empieza la clave).
- Estás escribiendo la clave: solo las letras de `name` (o de la siguiente clave
  esperada), en orden.
- Cerrada la clave `"name"` y puesto `:`, viene ` "` y luego **solo** caracteres
  que formen uno de los nombres válidos (`fn_add_numbers`, `fn_greet`, …). Si ya
  llevas `fn_g`, el único nombre compatible es `fn_greet`, así que el siguiente
  carácter obligatorio es `r`.
- Tras el nombre y su comilla de cierre: `,` y luego la clave `"parameters"`.
- Dentro de `parameters`: las claves son **exactamente** los parámetros de la
  función elegida, y los valores tienen que ser del tipo correcto:
  - `number` → dígitos, opcional `-` al principio, opcional un único `.`.
  - `string` → `"` … caracteres … `"` (con las reglas de escape de JSON).
  - `boolean` → solo `true` o `false`.
- Cuando todos los parámetros requeridos están puestos: `}` y `}` y fin.

### Ejercicio 8.1

Dibuja en papel el **diagrama de estados** para generar exactamente esta forma:

```
{"a": <number>}
```

Estados sugeridos: `INICIO`, `ESPERO_CLAVE`, `EN_CLAVE`, `ESPERO_DOSPUNTOS`,
`ESPERO_VALOR`, `EN_NUMERO`, `FIN`. Para cada estado, escribe:

- qué caracteres/tokens son válidos,
- a qué estado te llevan,
- si el estado es "final" (puedes parar).

### Ejercicio 8.2

Usa tu diagrama para decir, en cada una de estas salidas parciales, cuál es el
**conjunto de caracteres válidos** para el siguiente:

1. `` (vacío)
2. `{`
3. `{"a"`
4. `{"a":`
5. `{"a": 3`
6. `{"a": 3.`
7. `{"a": 3.14`
8. `{"a": 3.14}`

### Ejercicio 8.3

Amplía el diagrama del 8.1 para la forma real del proyecto:

```
{"name": "<nombre válido>", "parameters": {<pares clave:valor según la función>}}
```

usando **estas** dos funciones como universo posible:
`fn_greet(name: string)` y `fn_add_numbers(a: number, b: number)`.
Pista: en cuanto el nombre de función queda decidido, el resto del diagrama
(qué claves y qué tipos se permiten dentro de `parameters`) queda fijado.

---

## Parte 9 — El puente difícil: de "carácter válido" a "token válido"

Tu máquina de estados razona en **caracteres**. Pero el modelo elige **tokens**,
y un token puede ser varios caracteres a la vez (`"fn"`, `"_add"`, `"345"`…).
Este es el punto donde casi todo el mundo se atasca. Reglas:

1. Un token es **válido ahora** si, al añadir su texto a lo que llevas generado,
   **todo** ese texto nuevo sigue el camino permitido por la máquina de estados.
2. Para saber el texto de cada token usas el **vocabulario** (`vocab.json` +
   `merges.txt`), que mapea ID ↔ texto.
3. Un token puede "adelantar varios estados" de golpe. Ejemplo: si estás en
   `ESPERO_VALOR` y esperas un número, el token `"345"` es válido y te deja en
   `EN_NUMERO` habiendo escrito tres dígitos de una vez.
4. Un token es inválido si **cualquier** carácter suyo se sale del camino.
   Ejemplo: estás escribiendo el nombre y llevas `fn_g`; el token `"reet"` es
   válido (completa `fn_greet`), pero el token `"reeer"` no.
5. Hay que tener cuidado con los tokens que **incluyen el espacio o la comilla**
   (`Ġ"`, `":`), porque encajan o no según el estado exacto en el que estés.

En la práctica, en cada paso construyes un conjunto "IDs permitidos" mirando
todos (o los candidatos con logit alto) y comprobando la regla 1. A los demás
les pones el logit a −∞.

### Ejercicio 9.1

Mini-vocabulario inventado (ID → texto):

| ID | texto |
|---|---|
| 1 | `{` |
| 2 | `"` |
| 3 | `name` |
| 4 | `na` |
| 5 | `me` |
| 6 | `:` |
| 7 | ` ` (espacio) |
| 8 | `fn_greet` |
| 9 | `fn_` |
| 10 | `greet` |
| 11 | `add` |
| 12 | `}` |
| 13 | `x` |

Formato objetivo: `{"name": "fn_greet"}` (solo existe esa función).

Para cada salida parcial, lista **todos los IDs válidos** como siguiente token:

1. `` (vacío)
2. `{`
3. `{"`
4. `{"name`
5. `{"name": "`
6. `{"name": "fn_`
7. `{"name": "fn_greet`
8. `{"name": "fn_greet"`

(Nota cómo en el paso 4 tanto el ID 3 como… nada más, porque ya escribiste
`name` entero; y cómo en el paso 3 valen tanto `name` (ID 3) como `na` (ID 4).)

### Ejercicio 9.2

Con el mismo mini-vocabulario, imagina que en el paso 5 (`{"name": "`) el modelo
te da estos logits: `ID 8: 1.2`, `ID 9: 3.5`, `ID 11: 4.0`, `ID 13: 5.1`,
`ID 12: 2.0`. 

1. ¿Cuál elegiría un *greedy* **sin** restringir?
2. ¿Cuáles son los IDs válidos según el formato?
3. Tras poner los inválidos a −∞, ¿cuál se elige?
4. ¿Ves por qué esto lleva la fiabilidad del 30 % al ~100 %?

---

## Parte 10 — Encajar el esquema (no solo "JSON válido")

"JSON válido" no basta. El subject pide que la salida **cumpla el esquema** de
`functions_definition.json`:

- `name` tiene que ser **uno de los nombres definidos**, exactamente.
- Las claves dentro de `parameters` tienen que ser **exactamente** los parámetros
  de esa función (ni de más, ni de menos).
- Cada valor tiene que ser del **tipo** declarado (`number`, `string`, …).
- No se permite texto en prosa, ni claves extra, ni comas finales.

Es decir, tu máquina de estados de la Parte 8 **se genera a partir del fichero de
definiciones**: distintas funciones → distintos caminos permitidos.

### Ejercicio 10.1

Usando el `functions_definition.json` real del proyecto (5 funciones), escribe a
mano la **salida esperada** para cada uno de estos prompts del fichero de
pruebas:

1. `"What is the sum of 265 and 345?"`
2. `"Greet john"`
3. `"Reverse the string 'world'"`
4. `"What is the square root of 16?"`
5. `"Replace all vowels in 'Programming is fun' with asterisks"`

Formato de cada respuesta: `{"prompt": ..., "name": ..., "parameters": {...}}`
(mira el ejemplo de salida del subject, sección V.4.1). Presta atención al 5:
¿qué valor de `regex` pondrías? ¿y `replacement`? Aquí el modelo tiene que
**entender** la petición, no solo copiar números.

### Ejercicio 10.2

De la lista anterior, ¿en cuáles la decisión es "fácil" (basta con detectar la
función y copiar argumentos del texto) y en cuáles el modelo tiene que
**razonar / traducir** (p. ej. "todas las vocales" → un patrón regex)? Esto te
dice qué casos serán los frágiles al probar.

---

## Parte 11 — Juntarlo todo: plan mental del proyecto

Cuando entiendas las partes 1–10, el proyecto es esta tubería:

1. **Leer entradas:** `functions_definition.json` (esquema) y
   `function_calling_tests.json` (lista de prompts). Manejar con elegancia que el
   JSON venga roto o falte.
2. **Para cada prompt:**
   a. Construir el prompt de texto para el modelo (Parte 6): funciones + pregunta
      + instrucción de formato.
   b. `encode` → input IDs.
   c. **Bucle de generación con decodificación restringida** (Partes 5, 8, 9):
      - pedir logits con `get_logits_from_input_ids`,
      - calcular tokens permitidos con tu máquina de estados construida desde el
        esquema (Parte 10),
      - poner a −∞ los prohibidos,
      - elegir token, añadirlo, repetir,
      - parar cuando el objeto está completo y cierra bien.
   d. `decode` del resultado → texto JSON.
   e. Parsear ese JSON (será válido por construcción) → objeto.
3. **Escribir la salida:** un único fichero
   `data/output/function_calling_results.json` con un objeto por prompt:
   `{"prompt", "name", "parameters"}`.
4. **Errores con elegancia** en todo momento: nunca una excepción sin capturar,
   siempre un mensaje claro.

### Comprobación final de comprensión

Responde estas sin mirar atrás. Si dudas en alguna, vuelve a la parte indicada.

1. ¿Por qué no se puede resolver esto pidiéndole el JSON al modelo y ya? *(P.7)*
2. ¿Qué es exactamente un logit y qué significa ponerlo a −∞? *(P.4, P.8)*
3. ¿Por qué la máquina de estados razona en caracteres pero hay que traducirla a
   tokens, y por qué eso es lo difícil? *(P.9)*
4. ¿De dónde sacas el texto de cada token? *(P.3)*
5. ¿Qué pasa en el bucle de generación después de elegir un token? *(P.5)*
6. ¿Por qué "JSON válido" no es suficiente y qué añade "cumplir el esquema"? *(P.10)*
7. ¿Está permitido elegir la función con `if "sum" in prompt`? *(Subject IV.3.1:
   no — la función se elige **con el LLM**.)*
8. ¿Dónde has puesto el modelo descargado y con qué variable se lo dices a
   Hugging Face? *(P.0: `~/goinfre/hf_cache`, `HF_HOME`.)*

---

## Apéndice — `uv` en dos minutos

`uv` es un gestor de paquetes y entornos de Python (como `pip` + `venv`, pero
rápido y con fichero de bloqueo).

- `pyproject.toml` — lista de dependencias que quiere el proyecto.
- `uv.lock` — versiones exactas congeladas (reproducible).
- `uv sync` — lee esos dos ficheros y crea/actualiza un entorno virtual en
  `.venv/` con justo esas dependencias. Es lo único que ejecutará quien te
  corrija.
- `uv run <algo>` — ejecuta `<algo>` **dentro** de ese entorno, sin que tengas
  que "activarlo" a mano. Por eso el subject lanza el proyecto con
  `uv run python -m src ...`.
- Para `llm_sdk`: el subject dice que lo copies en el mismo directorio que `src`
  y lo añadas como dependencia local; tras `uv sync` estará disponible.

Orden típico la primera vez:

```
export HF_HOME=~/goinfre/hf_cache      # (ya en tu ~/.hellishrc, ver Parte 0)
uv sync
uv run hf download Qwen/Qwen3-0.6B     # descarga a goinfre
uv run python -m src                    # ejecuta tu proyecto
```
