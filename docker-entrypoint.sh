#!/bin/sh
# Starts Strata's own server (edge0-strata/serve/server.py) for the model folder named by the
# environment. It serves the OpenAI/Anthropic API and Strata's web page on the same port.
#
#   docker run ... IMAGE                     -> serve, configured from the environment
#   docker run ... IMAGE --parallel 2        -> same, with extra server.py flags appended
#   docker run ... IMAGE edge0-infer inspect --model /models/X --json   -> runs that command instead
set -euf   # -f: no glob expansion when splitting SERVE_ARGS

die() { echo "edge0: $*" >&2; exit 1; }

# A command that doesn't start with "-" replaces the server.
if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
    exec "$@"
fi

# ---- model folder ----
if [ -z "${MODEL_DIR:-}" ]; then
    [ -n "${EDGE0_MODEL:-}" ] || die "set EDGE0_MODEL (a folder name under /models) or MODEL_DIR (a full path)"
    MODEL_DIR="/models/${EDGE0_MODEL}"
    export MODEL_DIR
fi
[ -f "${MODEL_DIR}/model.json" ] || die "no model.json in ${MODEL_DIR}: mount the folder that holds the model files (see README)"
[ -w "${MODEL_DIR}" ] || echo "edge0: warning: ${MODEL_DIR} is read-only; the first start may need to write files into it" >&2

# ---- API key (passed to the server in STRATA_API_KEY, so it is not on the command line) ----
if [ -n "${API_KEY_FILE:-}" ]; then
    [ -r "${API_KEY_FILE}" ] || die "API_KEY_FILE=${API_KEY_FILE} is not readable"
    STRATA_API_KEY="$(cat "${API_KEY_FILE}")"
elif [ -n "${API_KEY:-}" ]; then
    STRATA_API_KEY="${API_KEY}"
else
    die "set API_KEY or API_KEY_FILE"
fi
[ -n "${STRATA_API_KEY}" ] || die "the API key is empty"
export STRATA_API_KEY
unset API_KEY

# ---- server config ----
umask 077
CONFIG=/run/edge0/strata.json
python3 /usr/local/bin/edge0-strata-config.py "${CONFIG}"

extra="$*"
cd "${EDGE0_STRATA_DIR}"
set -- "${EDGE0_STRATA_DIR}/.venv/bin/python" serve/server.py --engine strata --config "${CONFIG}" \
    --host "${HOST:-0.0.0.0}" --port "${PORT:-8080}"
# shellcheck disable=SC2086
[ -n "${extra}${SERVE_ARGS:-}" ] && set -- "$@" ${SERVE_ARGS:-} ${extra}

echo "edge0: $*" >&2
exec "$@"
