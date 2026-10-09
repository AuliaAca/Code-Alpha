#!/usr/bin/env bash
# One-time setup on macOS / Linux:  bash setup.sh
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
"$PY" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" \
  || { echo "Python 3.10 or newer is required"; exit 1; }
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
echo
echo "Setup complete. Start the web app with:  .venv/bin/python -m streamlit run app.py"
