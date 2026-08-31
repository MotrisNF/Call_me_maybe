#!/usr/bin/env python3
"""Pruebas manuales progresivas para el proyecto "call me maybe".

Este fichero NO es parte de la entrega (los programas de prueba no se
califican). Es una escalera: cada comprobación valida un peldaño del
proyecto. Ve haciéndolas en orden y no pases a la siguiente hasta que la
anterior imprima "OK".

Cómo se usa
-----------
    uv run python pruebas_manuales.py 1      # una comprobación
    uv run python pruebas_manuales.py 6
    uv run python pruebas_manuales.py all    # todas de una vez (recomendado)

Sin argumentos muestra este texto.

Sobre el modelo (léelo)
-----------------------
- La PRIMERA ejecución descarga Qwen3-0.6B (~1,2 GB). Necesitas internet
  esa vez y tarda.
- Queda en la caché de Hugging Face (~/.cache/huggingface/hub/), NO en
  el .venv. No se vuelve a descargar aunque borres el entorno virtual.
- En cada ejecución el modelo se carga en RAM desde esa caché (~10-30 s).
  Por eso "all" es más cómodo: carga una sola vez.
- Las comprobaciones 6 y 7 recorren todo el vocabulario en Python: son
  lentas a propósito (un minuto o así). Tu código de verdad usará numpy
  para eso.

Checklist
---------
[ ] 1  El SDK carga y encode()/decode() hacen ida y vuelta.
[ ] 2  get_logits_from_input_ids() da una puntuación por token.
[ ] 3  vocab.json se lee y sé pasar de nº de token a texto.
[ ] 4  Bucle "greedy" sin restricciones: veo qué contesta el modelo.
[ ] 5  Restricción mínima: fuerzo que el primer token sea "{".
[ ] 6  Restrinjo a los nombres de función: el modelo elige uno válido.
[ ] 7  Restrinjo el valor de un argumento a su tipo (p. ej. number).
[ ] 8  Mi programa (src/) genera data/output/... y pasa la validación.
[ ] 9  Errores con elegancia: fichero que falta / JSON roto, sin crash.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "data" / "input"
OUTPUT_FILE = ROOT / "data" / "output" / "function_calling_results.json"

# Importa llm_sdk tanto si está instalado (uv sync) como si solo lo has
# descomprimido al lado de este fichero. Si aún no está, no pasa nada:
# las comprobaciones 8 y 9 no lo necesitan; el resto avisará al ejecutarse.
Small_LLM_Model: Any = None
try:
    from llm_sdk import Small_LLM_Model  # type: ignore[no-redef]
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT / "llm_sdk"))
    try:
        from llm_sdk import Small_LLM_Model  # type: ignore[no-redef]
    except ModuleNotFoundError:
        pass

_MODEL: Any = None


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def modelo() -> Any:
    """Carga el modelo una sola vez por proceso."""
    global _MODEL
    if Small_LLM_Model is None:
        raise SystemExit(
            "No encuentro el paquete llm_sdk.\n"
            "Descomprime llm_sdk.zip aquí (quedará una carpeta llm_sdk/) "
            "o instálalo con 'uv sync', y vuelve a intentarlo.\n"
            "Las comprobaciones 8 y 9 sí funcionan sin él.")
    if _MODEL is None:
        print("Cargando Qwen3-0.6B (la 1ª vez descarga ~1,2 GB)...")
        _MODEL = Small_LLM_Model()
        print("Modelo listo.\n")
    return _MODEL


def encode(texto: str) -> list[int]:
    """Texto -> lista de números de token."""
    return modelo().encode(texto).tolist()[0]


def id_a_texto() -> dict[int, str]:
    """Diccionario nº de token -> cadena, a partir de vocab.json.

    Aviso: las cadenas usan codificación "byte-level" de BPE: 'Ġ' es un
    espacio delante, 'Ċ' un salto de línea, etc. Para ASCII normal
    (dígitos, '{', nombres tipo fn_greet) las claves son literales, que
    es lo único que necesitan estas pruebas.
    """
    ruta = modelo().get_path_to_vocab_file()
    with open(ruta, encoding="utf-8") as fichero:
        vocab: dict[str, int] = json.load(fichero)   # {"texto": numero}
    return {numero: texto for texto, numero in vocab.items()}


def argmax(valores: list[float]) -> int:
    return max(range(len(valores)), key=valores.__getitem__)


def cargar_funciones() -> list[dict]:
    ruta = INPUT_DIR / "functions_definition.json"
    with open(ruta, encoding="utf-8") as fichero:
        return json.load(fichero)


def cargar_pruebas() -> list[dict]:
    ruta = INPUT_DIR / "function_calling_tests.json"
    with open(ruta, encoding="utf-8") as fichero:
        return json.load(fichero)


# --------------------------------------------------------------------------
# 1. El SDK responde
# --------------------------------------------------------------------------
def check_1() -> None:
    m = modelo()
    ids = m.encode("Hello world").tolist()[0]
    print("IDs de 'Hello world':", ids)
    vuelta = m.decode(ids)
    print("decode(encode(x)) ->", repr(vuelta))
    assert "Hello world" in vuelta
    print("OK 1: el SDK funciona.")


# --------------------------------------------------------------------------
# 2. Logits: una puntuación por cada token posible
# --------------------------------------------------------------------------
def check_2() -> None:
    ids = encode("The capital of France is")
    logits = modelo().get_logits_from_input_ids(ids)
    print("nº de logits (= tamaño del vocabulario):", len(logits))
    mejor = argmax(logits)
    print("token siguiente más probable:", mejor,
          "->", repr(modelo().decode([mejor])))
    assert len(logits) > 1000
    print("OK 2: hay una puntuación por cada token del vocabulario.")


# --------------------------------------------------------------------------
# 3. Del número de token a su texto (vocab.json)
# --------------------------------------------------------------------------
def check_3() -> None:
    tabla = id_a_texto()
    print("tokens en el vocabulario:", len(tabla))
    con_espacio = [t for t in tabla.values() if t.startswith("Ġ")][:5]
    print("ejemplos con espacio delante (Ġ...):", con_espacio)
    ids = encode("fn_greet")
    print("'fn_greet' se escribe con estos tokens:",
          [(i, tabla.get(i)) for i in ids])
    print("OK 3: sé traducir número de token -> texto.")


# --------------------------------------------------------------------------
# 4. El bucle de generación, SIN restricciones
# --------------------------------------------------------------------------
def check_4() -> None:
    ids = encode("Question: what is 2 + 3? Answer:")
    generado = ""
    for _ in range(20):
        logits = modelo().get_logits_from_input_ids(ids)
        siguiente = argmax(logits)
        ids.append(siguiente)
        generado += modelo().decode([siguiente])
    print("El modelo, a su aire, continúa con:")
    print(repr(generado))
    print("OK 4: entiendo el bucle token a token "
          "(logits -> elegir -> añadir -> repetir).")


# --------------------------------------------------------------------------
# 5. Restricción mínima: forzar el primer token
# --------------------------------------------------------------------------
def check_5() -> None:
    tabla = id_a_texto()
    texto_a_id = {t: i for i, t in tabla.items()}
    abre = texto_a_id.get("{")
    print("el token de '{' es el número:", abre)
    assert abre is not None, "no encuentro '{' en el vocabulario"

    ids = encode("Give me JSON. Output:")
    logits = modelo().get_logits_from_input_ids(ids)
    # "tachamos" todo menos '{'
    forzado = [float("-inf")] * len(logits)
    forzado[abre] = logits[abre]
    elegido = argmax(forzado)
    print("primer token (forzado) ->", repr(tabla[elegido]))
    assert elegido == abre
    print("OK 5: sé poner logits a -inf para forzar/prohibir tokens.")


# --------------------------------------------------------------------------
# 6. Restringir la generación a un CONJUNTO de cadenas (los nombres)
# --------------------------------------------------------------------------
def generar_de_conjunto(prompt: str, objetivos: list[str],
                        tabla: dict[int, str], limite: int = 12) -> str:
    """Genera token a token permitiendo SOLO caminos que llevan a una de
    las cadenas de 'objetivos'. Versión de juguete: lenta y en Python."""
    ids = encode(prompt)
    generado = ""
    for _ in range(limite):
        if generado in objetivos:
            break
        logits = modelo().get_logits_from_input_ids(ids)
        mejor_id, mejor_val = -1, float("-inf")
        for tid, val in enumerate(logits):
            trozo = tabla.get(tid, "")
            if not trozo:
                continue
            candidato = generado + trozo
            encaja = any(o == candidato or o.startswith(candidato)
                         for o in objetivos)
            if encaja and val > mejor_val:
                mejor_id, mejor_val = tid, val
        if mejor_id < 0:
            break
        ids.append(mejor_id)
        generado += tabla[mejor_id]
    return generado


def check_6() -> None:
    funciones = cargar_funciones()
    nombres = [f["name"] for f in funciones]
    tabla = id_a_texto()
    catalogo = "\n".join(f'- {f["name"]}: {f["description"]}'
                         for f in funciones)
    print("(esto tarda ~1 min: recorre el vocabulario en Python)\n")
    for caso in cargar_pruebas()[:5]:
        peticion = caso["prompt"]
        prompt = (
            "You map a request to ONE function name from the list.\n"
            f"Functions:\n{catalogo}\n\n"
            f"Request: {peticion}\n"
            "Function name only: "
        )
        elegida = generar_de_conjunto(prompt, nombres, tabla)
        print(f"{peticion!r:60} -> {elegida!r}")
        # lo que garantiza la restricción: NUNCA sale algo imposible
        assert any(n == elegida or n.startswith(elegida) for n in nombres), \
            f"generó algo que no es prefijo de ningún nombre: {elegida!r}"
    print("\nOK 6: la salida SIEMPRE es (parte de) un nombre válido, "
          "aunque el modelo sea pequeño y a veces no acierte el sentido.")


# --------------------------------------------------------------------------
# 7. Restringir el VALOR de un argumento a su tipo
# --------------------------------------------------------------------------
def check_7() -> None:
    tabla = id_a_texto()
    prompt = ('Extract the first number.\n'
              'Text: "What is the sum of 265 and 345?"\n'
              'First number: ')
    ids = encode(prompt)
    logits = modelo().get_logits_from_input_ids(ids)

    def top5(valores: list[float]) -> list[str]:
        orden = sorted(range(len(valores)),
                       key=valores.__getitem__, reverse=True)
        return [repr(tabla.get(i, "")) for i in orden[:5]]

    print("Top-5 SIN restringir :", top5(logits))
    # dejamos solo tokens formados exclusivamente por dígitos
    mascara = list(logits)
    for tid, trozo in tabla.items():
        if not trozo or not all(c in "0123456789" for c in trozo):
            mascara[tid] = float("-inf")
    print("Top-5 SOLO dígitos   :", top5(mascara))
    elegido = argmax(mascara)
    assert tabla[elegido].isdigit()
    print("OK 7: puedo forzar que el valor respete el tipo del argumento "
          "(number aquí; igual para string / boolean).")


# --------------------------------------------------------------------------
# 8. Tu programa entero
# --------------------------------------------------------------------------
def validar_salida(resultados: list[dict], funciones: list[dict]) -> None:
    por_nombre = {f["name"]: f["parameters"] for f in funciones}
    for i, r in enumerate(resultados):
        assert set(r) == {"prompt", "name", "parameters"}, \
            f"[{i}] claves {set(r)} != prompt/name/parameters"
        assert r["name"] in por_nombre, \
            f"[{i}] función desconocida: {r['name']!r}"
        esperados = por_nombre[r["name"]]
        assert set(r["parameters"]) == set(esperados), \
            f"[{i}] argumentos {set(r['parameters'])} != {set(esperados)}"
        for clave, valor in r["parameters"].items():
            tipo = esperados[clave]["type"]
            if tipo == "number":
                ok = isinstance(valor, (int, float)) and not isinstance(
                    valor, bool)
            elif tipo == "string":
                ok = isinstance(valor, str)
            elif tipo == "boolean":
                ok = isinstance(valor, bool)
            else:
                ok = True
            assert ok, f"[{i}] {clave}={valor!r} no es {tipo}"


def check_8() -> None:
    if not (ROOT / "src" / "__main__.py").exists():
        print("Todavía no existe src/__main__.py.")
        print("Cuando lo tengas, esta comprobación ejecutará tu programa "
              "y validará data/output/function_calling_results.json contra "
              "el esquema de functions_definition.json.")
        return
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.unlink(missing_ok=True)
    cmd = [
        sys.executable, "-m", "src",
        "--functions_definition",
        str(INPUT_DIR / "functions_definition.json"),
        "--input", str(INPUT_DIR / "function_calling_tests.json"),
        "--output", str(OUTPUT_FILE),
    ]
    print("Ejecutando:", " ".join(cmd), "\n")
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout[-2000:])
    if res.returncode != 0:
        print("STDERR:\n", res.stderr[-2000:])
        raise SystemExit("tu programa terminó con código de error")
    assert OUTPUT_FILE.exists(), "no se creó el fichero de salida"
    with open(OUTPUT_FILE, encoding="utf-8") as fichero:
        resultados = json.load(fichero)      # si peta aquí, el JSON no vale
    validar_salida(resultados, cargar_funciones())
    print(f"OK 8: {len(resultados)} resultados y todos válidos "
          "contra el esquema.")


# --------------------------------------------------------------------------
# 9. Errores con elegancia
# --------------------------------------------------------------------------
def check_9() -> None:
    if not (ROOT / "src" / "__main__.py").exists():
        print("Todavía no existe src/. Cuando lo tengas, prueba a mano:")
        print("  uv run python -m src --input no_existe.json")
        print("  printf '{ roto' > /tmp/malo.json && "
              "uv run python -m src --input /tmp/malo.json")
        print("Debe: mensaje claro para el usuario, sin 'Traceback', "
              "y terminar sin caerse.")
        return
    roto = ROOT / "_entrada_rota.json"
    roto.write_text("{ esto no es json valido", encoding="utf-8")
    casos = [
        ["--input", "no_existe_seguro_12345.json"],
        ["--input", str(roto)],
    ]
    try:
        for extra in casos:
            res = subprocess.run(
                [sys.executable, "-m", "src", *extra],
                cwd=ROOT, capture_output=True, text=True)
            salida = res.stdout + res.stderr
            print(f"args {extra} -> returncode {res.returncode}")
            print("  dice:", salida.strip()[:200] or "(nada)")
            assert "Traceback" not in salida, \
                "lanzó una excepción sin capturar (hay Traceback)"
    finally:
        roto.unlink(missing_ok=True)
    print("OK 9: entradas malas -> mensaje claro y sin crash.")


# --------------------------------------------------------------------------
CHECKS: dict[int, Callable[[], None]] = {
    1: check_1, 2: check_2, 3: check_3, 4: check_4, 5: check_5,
    6: check_6, 7: check_7, 8: check_8, 9: check_9,
}


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    if args[0] == "all":
        for numero in sorted(CHECKS):
            print(f"\n===== comprobación {numero} =====")
            try:
                CHECKS[numero]()
            except AssertionError as error:
                print(f"FALLA {numero}: {error}")
            except Exception as error:            # noqa: BLE001
                print(f"ERROR {numero}: {type(error).__name__}: {error}")
        return
    try:
        numero = int(args[0])
    except ValueError:
        print(f"argumento no válido: {args[0]!r} (usa 1..9 o 'all')")
        return
    if numero not in CHECKS:
        print(f"no hay comprobación {numero} (usa 1..9 o 'all')")
        return
    CHECKS[numero]()


if __name__ == "__main__":
    main()
