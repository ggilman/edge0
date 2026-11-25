#!/bin/sh
# Turns the container's environment variables into an `edge0-infer serve` command line.
#
#   docker run ... IMAGE                       -> serve, configured from the environment
#   docker run ... IMAGE --temperature 0.6     -> same, with extra serve flags appended
#   docker run ... IMAGE edge0-infer inspect --model /models/X --json   -> runs that command instead
set -euf   # -f: no glob expansion when splitting MODEL_OPTIONS / STRATA_ARGS / SERVE_ARGS

die() { echo "edge0: $*" >&2; exit 1; }

# A command that doesn't start with "-" replaces the server.
if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
    exec "$@"
fi

# ---- model folder ----
if [ -z "${MODEL_DIR:-}" ]; then
    [ -n "${EDGE0_MODEL:-}" ] || die "set EDGE0_MODEL (a folder name under /models) or MODEL_DIR (a full path)"
    MODEL_DIR="/models/${EDGE0_MODEL}"
fi
[ -f "${MODEL_DIR}/model.json" ] || die "no model.json in ${MODEL_DIR}: mount the folder that holds the model files (see README)"
[ -w "${MODEL_DIR}" ] || echo "edge0: warning: ${MODEL_DIR} is read-only; the first start may need to write files into it" >&2

# ---- API key ----
if [ -n "${API_KEY_FILE:-}" ]; then
    [ -r "${API_KEY_FILE}" ] || die "API_KEY_FILE=${API_KEY_FILE} is not readable"
    KEY_FILE="${API_KEY_FILE}"
elif [ -n "${API_KEY:-}" ]; then
    umask 077
    mkdir -p /run/edge0
    KEY_FILE=/run/edge0/api-key
    printf '%s' "${API_KEY}" > "${KEY_FILE}"
else
    # Without a key the server only answers requests from inside the container (loopback),
    # so a published port would refuse every client.
    die "set API_KEY or API_KEY_FILE"
fi

# ---- build the serve command ----
extra="$*"
set -- edge0-infer serve --model "${MODEL_DIR}" --host "${HOST:-0.0.0.0}" --port "${PORT:-8080}" \
    --api-key-file "${KEY_FILE}"
[ -n "${CONTEXT:-}" ]         && set -- "$@" --context "${CONTEXT}"
[ -n "${DEVICE:-}" ]          && set -- "$@" --device "${DEVICE}"
[ -n "${MODEL_ALIAS:-}" ]     && set -- "$@" --alias "${MODEL_ALIAS}"
[ -n "${THINKING:-}" ]        && set -- "$@" --thinking "${THINKING}"
[ -n "${READY_TIMEOUT:-}" ]   && set -- "$@" --strata-ready-timeout "${READY_TIMEOUT}"
for opt in ${MODEL_OPTIONS:-}; do set -- "$@" --model-option "${opt}"; done
for word in ${STRATA_ARGS:-}; do set -- "$@" --strata-arg "${word}"; done
# shellcheck disable=SC2086
[ -n "${extra}${SERVE_ARGS:-}" ] && set -- "$@" ${SERVE_ARGS:-} ${extra}

echo "edge0: $*" | sed 's/--api-key-file [^ ]*/--api-key-file <hidden>/' >&2
exec "$@"
