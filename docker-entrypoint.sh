#!/bin/sh
set -e

for secret in api_key langsmith_api_key; do
    file="/run/secrets/${secret}"
    if [ -f "$file" ]; then
        var=$(echo "$secret" | tr '[:lower:]' '[:upper:]')
        export "$var"="$(cat "$file")"
    fi
done

exec "$@"
