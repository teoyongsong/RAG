#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run.sh setup
#   ./run.sh streamlit
#   ./run.sh web
#   ./run.sh ingest
#   ./run.sh ask "What is supervised learning?"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

ensure_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
}

setup_env() {
  ensure_venv
  echo "Installing local dependencies..."
  pip install -r "${ROOT_DIR}/requirements_local.txt"
  echo "Setup complete."
}

run_streamlit() {
  ensure_venv
  # Local run should expose full controls unless user overrides explicitly.
  export PUBLIC_DEMO_MODE="${PUBLIC_DEMO_MODE:-0}"
  streamlit run "${ROOT_DIR}/streamlit_app.py"
}

run_web() {
  ensure_venv
  python "${ROOT_DIR}/web_app.py"
}

run_ingest() {
  ensure_venv
  python "${ROOT_DIR}/ingest.py"
}

run_ask() {
  ensure_venv
  if [[ $# -lt 1 ]]; then
    echo "Please provide a question."
    echo 'Example: ./run.sh ask "What is time series forecasting?"'
    exit 1
  fi
  python "${ROOT_DIR}/rag.py" "$*"
}

print_help() {
  cat <<'EOF'
RAG Runner

Commands:
  setup                Create venv and install requirements_local.txt
  streamlit            Run Streamlit app
  web                  Run FastAPI web app
  ingest               Build/rebuild vector index
  ask "<question>"     Ask from CLI
EOF
}

cmd="${1:-help}"
shift || true

case "${cmd}" in
  setup) setup_env ;;
  streamlit) run_streamlit ;;
  web) run_web ;;
  ingest) run_ingest ;;
  ask) run_ask "$@" ;;
  help|-h|--help) print_help ;;
  *)
    echo "Unknown command: ${cmd}"
    print_help
    exit 1
    ;;
esac

