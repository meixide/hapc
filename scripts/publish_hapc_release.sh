#!/usr/bin/env bash
# HAPC release helper — wraps the usual git + make release + PyPI smoke flow.
#
# Typical sequence (from the repository root, or any path — the script cds):
#
#   1. Commit your work:
#        ./scripts/publish_hapc_release.sh commit "Describe your changes"
#
#   2. After CI is green, cut a release (bumps pyproject + __init__, tags vX.Y.Z, pushes):
#        ./scripts/publish_hapc_release.sh release 0.3.1 "Short release notes here."
#
#   3. After GitHub Actions finishes (~20–30 min), verify PyPI:
#        ./scripts/publish_hapc_release.sh smoke 0.3.1
#
# Or show this help:
#        ./scripts/publish_hapc_release.sh help

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
HAPC release helper — usual git + make release + PyPI smoke flow.

Typical sequence (run from anywhere; script cds to repo root):

  1. Commit your work:
       ./scripts/publish_hapc_release.sh commit "Describe your changes"

  2. After CI is green, cut a release (bumps version, tags vX.Y.Z, pushes):
       ./scripts/publish_hapc_release.sh release 0.3.1 "Short release notes here."

  3. After GitHub Actions finishes (~20–30 min), verify PyPI:
       ./scripts/publish_hapc_release.sh smoke 0.3.1

Commands:
  help                          Show this help.
  commit <message>              git add . && git commit -m "..." && git push
  release <version> [message]   make release VERSION=... [MSG=...]
  smoke <version>               make smoke-pypi VERSION=...
EOF
}

cmd="${1:-help}"
shift || true

case "${cmd}" in
  help|-h|--help)
    usage
    ;;
  commit)
    msg="${1:?commit message required, e.g.: $0 commit \"Your message\"}"
    git add .
    git commit -m "${msg}"
    git push
    echo ">>> Committed and pushed. When CI is green, run:"
    echo "    ./scripts/publish_hapc_release.sh release <new_version> \"release notes\""
    ;;
  release)
    ver="${1:?version required, e.g. 0.3.1}"
    shift || true
    msg="${*:-}"
    if [[ -n "${msg}" ]]; then
      make release "VERSION=${ver}" "MSG=${msg}"
    else
      make release "VERSION=${ver}"
    fi
    echo ">>> Tag pushed. Watch: https://github.com/meixide/hapc/actions"
    echo ">>> When done, verify PyPI:"
    echo "    ./scripts/publish_hapc_release.sh smoke ${ver}"
    ;;
  smoke)
    ver="${1:?version required, e.g. 0.3.1}"
    make smoke-pypi "VERSION=${ver}"
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage >&2
    exit 2
    ;;
esac
