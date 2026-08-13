#!/bin/sh
set -eu

ensure_screener_writable() {
    target=$1
    mkdir -p "$target"
    if runuser -u screener -- sh -c '
        probe="$1/.screener-write-test-$$"
        : > "$probe" && rm -f "$probe"
    ' sh "$target" 2>/dev/null; then
        return
    fi

    echo "Repairing screener ownership for $target" >&2
    chown -R --no-dereference screener:screener "$target"
    chmod -R u+rwX "$target"
    if ! runuser -u screener -- test -w "$target"; then
        echo "Cannot make $target writable by the screener user" >&2
        exit 1
    fi
}

ensure_screener_writable "${SCREENER_CACHE_DIR:-/app/cache}"
ensure_screener_writable "${SCREENER_OUTPUT_DIR:-/app/outputs}"

exec runuser -u screener -- "$@"
