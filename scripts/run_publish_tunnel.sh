#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HOST="${TDR_HOST:-127.0.0.1}"
PORT="${TDR_PORT:-8000}"
FRONTEND_CONFIG="${FRONTEND_CONFIG:-webapp/frontend/runtime-config.js}"
LOG_DIR="${LOG_DIR:-logs}"
BACKEND_PID_FILE="$LOG_DIR/uvicorn.pid"
CLOUDFLARED_PID_FILE="$LOG_DIR/cloudflared.pid"
BACKEND_OUT_LOG="$LOG_DIR/uvicorn.out.log"
BACKEND_ERR_LOG="$LOG_DIR/uvicorn.err.log"
CLOUDFLARED_OUT_LOG="$LOG_DIR/cloudflared.out.log"
CLOUDFLARED_ERR_LOG="$LOG_DIR/cloudflared.err.log"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Update backend tunnel}"
PUSH="${PUSH:-1}"
CLOUDFLARED_PROTOCOLS="${CLOUDFLARED_PROTOCOLS:-${CLOUDFLARED_PROTOCOL:-http2 quic}}"
TUNNEL_ATTEMPTS="${TUNNEL_ATTEMPTS:-5}"
TUNNEL_URL_ATTEMPTS="${TUNNEL_URL_ATTEMPTS:-45}"
TUNNEL_HEALTH_ATTEMPTS="${TUNNEL_HEALTH_ATTEMPTS:-60}"

mkdir -p "$LOG_DIR"

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

stop_pid_file() {
  local pid_file="$1"
  local label="$2"

  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      echo "Stopping existing $label process: $pid"
      kill "$pid" >/dev/null 2>&1 || true
      sleep 1
    fi
  fi
}

stop_port_listener() {
  local port="$1"
  local label="$2"
  local pids

  pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    return 0
  fi

  echo "Stopping existing $label listener on port $port: $pids"
  while IFS= read -r pid; do
    [[ -n "$pid" ]] && kill "$pid" >/dev/null 2>&1 || true
  done <<< "$pids"
  sleep 1
}

wait_for_health() {
  local url="$1"
  local label="$2"
  local attempts="${3:-30}"

  for _ in $(seq 1 "$attempts"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "$label is reachable: $url"
      return 0
    fi
    sleep 1
  done

  echo "ERROR: $label did not become reachable: $url" >&2
  return 1
}

find_python() {
  if [[ -n "${TDR_PYTHON:-}" ]]; then
    echo "$TDR_PYTHON"
    return
  fi

  if [[ -x ".venv/bin/python" ]]; then
    echo ".venv/bin/python"
    return
  fi

  if command_exists python3; then
    echo "python3"
    return
  fi

  if command_exists python; then
    echo "python"
    return
  fi

  echo "ERROR: Python not found. Set TDR_PYTHON=/path/to/python." >&2
  return 1
}

if ! command_exists cloudflared; then
  echo "ERROR: cloudflared not found in PATH." >&2
  exit 1
fi

PYTHON_BIN="$(find_python)"

export TDR_RESULTS_ROOT="${TDR_RESULTS_ROOT:-$HOME/Desktop/gw_tdr_results}"
export TDR_GWOSC_CACHE_ROOT="${TDR_GWOSC_CACHE_ROOT:-$TDR_RESULTS_ROOT/_gwosc_cache}"

stop_pid_file "$BACKEND_PID_FILE" "uvicorn"
stop_pid_file "$CLOUDFLARED_PID_FILE" "cloudflared"
stop_port_listener "$PORT" "backend"

echo "Starting backend on http://$HOST:$PORT"
nohup "$PYTHON_BIN" -m uvicorn webapp.backend.main:app \
  --host "$HOST" \
  --port "$PORT" \
  > "$BACKEND_OUT_LOG" \
  2> "$BACKEND_ERR_LOG" \
  < /dev/null &
echo $! > "$BACKEND_PID_FILE"

wait_for_health "http://127.0.0.1:$PORT/api/health" "Backend" 30

echo "Starting Cloudflare quick tunnel"
: > "$CLOUDFLARED_OUT_LOG"
: > "$CLOUDFLARED_ERR_LOG"
TUNNEL_URL=""
attempt=1

while [[ "$attempt" -le "$TUNNEL_ATTEMPTS" && -z "$TUNNEL_URL" ]]; do
  for protocol in $CLOUDFLARED_PROTOCOLS; do
    echo "Cloudflare attempt $attempt/$TUNNEL_ATTEMPTS using protocol $protocol"
    {
      echo
      echo "===== attempt $attempt protocol $protocol $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
    } >> "$CLOUDFLARED_OUT_LOG"
    {
      echo
      echo "===== attempt $attempt protocol $protocol $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
    } >> "$CLOUDFLARED_ERR_LOG"

    nohup cloudflared tunnel --protocol "$protocol" --url "http://127.0.0.1:$PORT" \
      >> "$CLOUDFLARED_OUT_LOG" \
      2>> "$CLOUDFLARED_ERR_LOG" \
      < /dev/null &
    echo $! > "$CLOUDFLARED_PID_FILE"

    candidate_url=""
    for _ in $(seq 1 "$TUNNEL_URL_ATTEMPTS"); do
      candidate_url="$(
        grep -hEo 'https://[A-Za-z0-9-]+\.trycloudflare\.com' "$CLOUDFLARED_OUT_LOG" "$CLOUDFLARED_ERR_LOG" 2>/dev/null \
          | tail -n 1 || true
      )"
      if [[ -n "$candidate_url" ]]; then
        break
      fi
      sleep 1
    done

    if [[ -z "$candidate_url" ]]; then
      echo "No trycloudflare URL found for this attempt."
      stop_pid_file "$CLOUDFLARED_PID_FILE" "cloudflared"
      continue
    fi

    if ! kill -0 "$(cat "$CLOUDFLARED_PID_FILE")" >/dev/null 2>&1; then
      echo "cloudflared exited early for $candidate_url"
      stop_pid_file "$CLOUDFLARED_PID_FILE" "cloudflared"
      continue
    fi

    if wait_for_health "$candidate_url/api/health" "Tunnel" "$TUNNEL_HEALTH_ATTEMPTS"; then
      TUNNEL_URL="$candidate_url"
      break
    fi

    echo "Tunnel URL did not become reachable, retrying with a fresh tunnel."
    stop_pid_file "$CLOUDFLARED_PID_FILE" "cloudflared"
  done

  attempt=$((attempt + 1))
done

if [[ -z "$TUNNEL_URL" ]]; then
  echo "ERROR: no Cloudflare quick tunnel became reachable." >&2
  echo "Check $CLOUDFLARED_OUT_LOG and $CLOUDFLARED_ERR_LOG" >&2
  exit 1
fi

cat > "$FRONTEND_CONFIG" <<EOF
window.TDR_WEB_CONFIG = {
  publicApiBase: "$TUNNEL_URL",
};
EOF
