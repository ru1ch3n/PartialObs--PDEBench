#!/usr/bin/env bash
set -Eeuo pipefail

: "${PDEOBS_REPO:?set PDEOBS_REPO}"
: "${PDEOBS_DATA:?set PDEOBS_DATA}"
: "${PDEOBS_RUN:?set PDEOBS_RUN}"
: "${PDEOBS_PYTHON:?set PDEOBS_PYTHON}"

mkdir -p "$PDEOBS_RUN/python" "$PDEOBS_RUN/mpl-cache" "$PDEOBS_RUN/output"

if ! PYTHONPATH="$PDEOBS_RUN/python" "$PDEOBS_PYTHON" -c 'import matplotlib' >/dev/null 2>&1; then
  "$PDEOBS_PYTHON" -m pip install \
    --disable-pip-version-check --no-input \
    --target "$PDEOBS_RUN/python" \
    'matplotlib>=3.7,<4'
fi

export PYTHONPATH="$PDEOBS_RUN/python:$PDEOBS_REPO/src"
export MPLCONFIGDIR="$PDEOBS_RUN/mpl-cache"

"$PDEOBS_PYTHON" "$PDEOBS_RUN/render_seawulf_figures.py" \
  --data-root "$PDEOBS_DATA" \
  --output-dir "$PDEOBS_RUN/output"

test -s "$PDEOBS_RUN/output/fig_observation_protocols.pdf"
test -s "$PDEOBS_RUN/output/fig_pde_gallery.pdf"
test -s "$PDEOBS_RUN/output/fig_full_quality_snapshot.pdf"
test -s "$PDEOBS_RUN/output/figure_snapshot.json"
