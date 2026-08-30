#!/bin/bash -eu

cd "$SRC/pdeobs"
export PYTHONPATH="$SRC/pdeobs/src"

compile_python_fuzzer "$SRC/pdeobs/.clusterfuzzlite/fuzz_config.py"
python3 -m zipfile -c "$OUT/fuzz_config_seed_corpus.zip" \
  "$SRC/pdeobs/.clusterfuzzlite/corpus/fuzz_config/valid_override.txt" \
  "$SRC/pdeobs/.clusterfuzzlite/corpus/fuzz_config/nested_override.txt" \
  "$SRC/pdeobs/.clusterfuzzlite/corpus/fuzz_config/environment_override.txt"
cp "$SRC/pdeobs/.clusterfuzzlite/fuzz_config.options" "$OUT/fuzz_config.options"
