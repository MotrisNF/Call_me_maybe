*This project has been created as part of the 42 curriculum by saperez-.*

# call me maybe

Turn a natural-language request into a **structured, schema-valid function
call** using a tiny local LLM (`Qwen/Qwen3-0.6B`) and **constrained decoding**.

## Description

Large language models are good at understanding language but unreliable at
emitting machine-readable output: asked for JSON, a 0.6B model produces a
*valid* object only a fraction of the time. This project fixes that by
controlling the model's output token by token so that the result is **always**
parseable and **always** matches the schema declared in
`data/input/functions_definition.json`.

Given, for example, `"What is the sum of 2 and 3?"`, the program does not answer
`5`. It produces:

```json
{ "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": { "a": 2.0, "b": 3.0 } }
```

The model still decides *which* function and *which* argument values; the
constraint only removes every token that would break the JSON or the schema.

### Project layout

| Path | Role |
|------|------|
| `src/vocab.py` | token id ↔ text tables, built from the SDK's public `decode` |
| `src/schema.py` | pydantic models: validate the catalogue, validate a generated call |
| `src/prompt.py` | build the chat prompt (function catalogue + few-shot examples) |
| `src/constrained.py` | the constrained decoder |
| `src/pipeline.py` | run every prompt, collect results, isolate per-prompt failures |
| `src/output.py` | write `data/output/function_calling_results.json` |
| `src/__main__.py` | CLI and orchestration |

## Instructions

### Requirements

* Python **3.10+**
* [`uv`](https://docs.astral.sh/uv/) for dependency management
* The provided `llm_sdk/` package (already vendored in this repository)

The reviewer and the moulinette only run `uv sync`. All dependencies
(`numpy`, `pydantic`, `torch` CPU build, plus the SDK) are pinned in
`pyproject.toml` / `uv.lock`.

### Install

```sh
make install        # runs `uv sync`
```

`torch` is pinned to its **CPU-only** build (`[tool.uv.sources]` in
`pyproject.toml`) so `uv sync` does not pull several GB of unused CUDA wheels.

On the first run the model weights (~1.5 GB) are downloaded into the Hugging
Face cache (`$HF_HOME`, default `~/.cache/huggingface`); later runs are offline.

> On the author's 42 workstation the home partition is ~2 GB. Run **`make
> install`** once before use: besides `uv sync`, it points `.venv` at a venv
> on the local scratch partition, so the plain `uv sync` / `uv run python -m
> src` a reviewer types afterwards also stay off the small partition (`$HF_HOME`
> and `$UV_CACHE_DIR` are set from `~/.hellishrc`). On any other machine this
> does nothing and a plain `uv sync` is all that is needed.

### Run

```sh
# default paths: reads data/input/, writes data/output/
uv run python -m src

# explicit paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Makefile targets: `install`, `run`, `debug` (under `pdb`), `lint`
(`flake8` + `mypy` with the required flags), `lint-strict` (`mypy --strict`),
`test` (`pytest`), `clean`, `fclean`, `re`.

## Example usage

```
$ make run
functions: 5  prompts: 11
wrote 11 call(s) to data/output/function_calling_results.json

$ cat data/output/function_calling_results.json
[
  { "prompt": "What is the sum of 265 and 345?",
    "name": "fn_add_numbers",
    "parameters": { "a": 265.0, "b": 345.0 } },
  { "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": { "name": "shrek" } },
  { "prompt": "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'",
    "name": "fn_substitute_string_with_regex",
    "parameters": { "source_string": "The cat sat on the mat with another cat",
                    "regex": "cat", "replacement": "dog" } }
  ...
]
```

A prompt that matches no catalogue function is skipped with a warning on
stderr and produces no entry, so every `name` in the output is a real
catalogue function.

## Algorithm — constrained decoding

### The generation loop

An LLM only predicts the **next token**. Text is produced by repeating:

```
tokenize prompt -> input ids
loop:
    logits = model.get_logits_from_input_ids(input_ids)   # one score per vocab token
    next_id = argmax(logits)                              # greedy choice
    input_ids.append(next_id)
```

Constrained decoding adds one step: **before `argmax`, drop every token that
would break the target JSON** (conceptually, set its logit to `-inf`). What
remains can only ever extend a valid output.

### Walking the fixed output shape

The subject fixes the output shape:

```
{"name": "<catalogue name>", "parameters": {"<key>": <value>, ...}}
```

So instead of a general JSON grammar, `generate_call` walks that shape
directly. The **fixed parts are injected** (their token ids are appended with
no model call — there is nothing to choose), and only the **holes** are
generated:

```
inject  {"name": "
generate  <function name>
inject  , "parameters": {
for each parameter:
    inject  "<key>":                (prefixed with ", " after the first)
    generate  <value>
    inject  , "  or  }}
```

This keeps the number of forward passes minimal (~6–12 for a simple call).

### Constraining each hole

* **Function name** (`_gen_choice`) — the output must become one of
  `["fn_add_numbers", ..., "none"]`. At each step only tokens that keep the
  string a prefix of some candidate are allowed. As soon as a single candidate
  remains, the rest is injected (no further model calls).

* **Number** (`_gen_number`) — only the ten digit tokens, a leading `-`, and a
  single `.` (for `number`, not `integer`) are allowed. Qwen tokenizes digits
  individually, so this set is tiny. The value ends when the model chooses to
  emit the terminator's first token, which is offered alongside the digits once
  the number is well-formed.

* **String** (`_gen_value` → `_gen_string_span` / `_gen_string_free`) — a
  function argument is *usually* a literal span copied from the request, so the
  body is first constrained to be a **contiguous substring of the request**.
  The model then physically cannot hallucinate a value. If, on the first
  character, the model would clearly rather emit something that is not a span
  (e.g. `[` to start a regex), the span mode bails and a structure-only mode
  takes over: any token that keeps the string a valid JSON string (a 4-state
  machine with escape handling), capped in length.

* **Boolean** — same mechanism as the name, over `["true", "false"]`.

Character-level rules meet token-level generation by *folding*: a token is
legal from a state iff feeding its characters one by one never leaves the
state machine.

### Why not a general JSON grammar

The output shape is not arbitrary — it is dictated by the subject and by the
catalogue. A specialised walker is smaller, faster (fewer forward passes,
constant-time constraint checks), and far easier to reason about and defend
than a general grammar engine.

## Design decisions

* **Token text via `decode`, not `vocab.json`.** `Vocab` calls the SDK's
  public `decode` once per id. Rebuilding the byte-level tokenizer from
  `vocab.json` + `merges.txt` is a bonus objective and would only add risk in
  the mandatory part. No private SDK attribute is touched anywhere.
* **Inject fixed literals.** The model is never asked to reproduce
  `{"name": "` or `, "parameters": {`; those ids are appended directly. Only
  genuine decisions cost a forward pass.
* **pydantic everywhere.** `ParamSpec` / `FunctionDef` validate the catalogue
  file; `params_model` uses `pydantic.create_model` to build, per function, a
  model that checks the generated `parameters` has exactly the right keys and
  value types. Constrained decoding already guarantees this structurally — the
  pydantic pass is a second, independent check and satisfies the "all classes
  use pydantic" rule.
* **Per-prompt isolation.** `process_prompts` wraps each prompt in
  `try/except`; a dead-end or a validation failure on one prompt produces a
  stderr warning and is skipped, never a crash.
* **`"none"` sentinel.** The name grammar also allows `"none"` so the model
  can decline instead of being forced into a wrong function. When it does, the
  prompt is skipped (no output row) — the output never carries a `name`
  outside the catalogue.
* **CPU-only `torch`.** The workstation has no NVIDIA GPU; pinning the CPU
  wheel keeps the environment ~1 GB instead of ~4 GB and pulls no `nvidia-*`
  packages.

## Performance analysis

Measured on the provided `function_calling_tests.json` (11 prompts):

| Metric | Result |
|--------|--------|
| JSON validity | **11 / 11** — valid by construction |
| Schema conformance (keys, types) | **11 / 11** |
| Function selection | **11 / 11** |
| Argument values correct | **18 / 19** (the one miss: "all vowels" was inferred as the regex `aeiouAEIOU` instead of `[aeiou]`) |
| Forward passes, whole run | ~130 (6–12 per simple call, 20–30 for regex substitutions) |
| Wall time, unloaded CPU | ~1–2 minutes (~0.3 s / forward + model load) |
| Wall time, heavily-loaded shared CPU | 4–5 minutes |

The program's own overhead (constraint checks, argmax over the restricted set,
substring lookups) is negligible — essentially 100 % of the time is
`get_logits_from_input_ids`. The SDK recomputes the whole prefix on every call
(no KV cache is exposed), which is the main cost and exactly what the bonus
"caching / batching" objective targets.

## Challenges faced

* **Digit tokenization.** Qwen has no multi-digit tokens, so numbers are built
  one digit at a time; the number constraint is therefore just "digits, one
  optional `-`, one optional `.`".
* **Knowing when a value ends.** A number or boolean has no closing delimiter
  of its own. Solved by offering the *next* literal's first token as a "stop"
  option once the value is well-formed.
* **Characters vs tokens.** The rules are naturally character-level but the
  model emits multi-character tokens. Solved by folding the state machine over
  a token's characters and only allowing tokens that fold cleanly.
* **Strings rambling.** With a permissive string constraint, greedy decoding of
  a 0.6B model wanders (`"description of the name"`). Solved by constraining
  string values to spans of the request, with an escape hatch for values that
  must be inferred.
* **No `eos_token_id`.** It lives on a private SDK attribute. Not needed in the
  end: the walker knows the JSON is complete when the final `}}` is injected.
* **Environment.** The 42 workstation's home partition is ~2 GB — too small for
  the `torch` install plus the 1.5 GB model. `~/.hellishrc` sets `HF_HOME` to
  the persistent network partition (`sgoinfre`, where the model lives, so it is
  never re-downloaded) and `UV_CACHE_DIR` to the fast local one (`goinfre`);
  `make install` makes `.venv` a symlink to a venv on `goinfre`. Earlier
  attempts symlinked the `~/.cache` directories directly and kept breaking — a
  scratch wipe left the symlinks dangling and tools silently recreated them as
  real directories on the full partition. Routing through env vars plus a
  single project-local `.venv` symlink is the version that stayed fixed. None
  of this is in the repo; another machine just runs `uv sync`.

## Testing strategy

* **Unit tests** (`tests/`, run with `make test`) drive the pure logic —
  schema loading and validation, the JSON-string state machine, the name/value
  grammars, and the output writer — with a hand-written fake model, so they
  need no weights and run in milliseconds.
* **Error paths** were exercised by hand: missing input file, malformed JSON,
  a non-array input, a catalogue entry missing `name`. Each produces a clear
  `error: ...` message and a non-zero exit, never a traceback.
* **End-to-end**: `make run` against the provided test set, with the results
  table above checked by hand against the expected calls.
* **Static analysis**: `make lint` and `make lint-strict` (i.e. `flake8 .` and
  `mypy .` / `mypy --strict`) are clean.

## Resources

### References

* Hugging Face — *Text generation strategies* (greedy vs sampling):
  <https://huggingface.co/docs/transformers/generation_strategies>
* Hugging Face — *Byte-Pair Encoding tokenization*:
  <https://huggingface.co/learn/nlp-course/chapter6/5>
* Qwen3 documentation (chat template, thinking mode):
  <https://qwenlm.github.io/blog/qwen3/>
* *Efficient Guided Generation for Large Language Models* (Willard & Louf,
  2023) — the theory behind masking logits against a grammar:
  <https://arxiv.org/abs/2307.09702>
* JSON grammar: <https://www.json.org/json-en.html>
* pydantic v2 docs — dynamic models: <https://docs.pydantic.dev/latest/concepts/models/#dynamic-model-creation>
* `uv` documentation: <https://docs.astral.sh/uv/>

### Use of AI

An AI assistant (Claude Code) was used, always with review of every line, for:

* **Environment setup** — redirecting the `uv` environment and the Hugging Face
  cache off the small home partition, pinning the CPU-only `torch` build, and
  writing the `Makefile` and `.flake8`.
* **Learning** — explaining the tokenize → logits → sample loop, BPE tokens,
  logits, and the idea of masking logits for constrained decoding, before any
  code was written.
* **Debugging** — diagnosing why string values rambled and why a prompt
  instruction alone cannot change a constrained field.

The prompt text in `src/prompt.py` and the accuracy/latency trade-offs were
tuned by hand against the provided test set.
