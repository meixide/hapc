#!/usr/bin/env bash
# scripts/release.sh — bump version, commit, tag, push (triggers PyPI publish).
#
# Usage:
#   scripts/release.sh <new_version> [-m "<release notes>"]
#
# Example:
#   scripts/release.sh 0.3.1 -m "Add ate_hapc + diagnostics."
#
# What this does, in order:
#   1. Validates the new version string (PEP 440-ish: X.Y.Z[.devN|aN|bN|rcN]).
#   2. Refuses to run if the working tree is dirty (other than the two version
#      files this script edits).
#   3. Refuses to run if the new version <= the current version.
#   4. Updates `pyproject.toml` and `python/hapc/__init__.py` in lock-step.
#   5. Runs `python -m pytest -q` (skipping the R-vs-Python integration test).
#   6. Commits with message "Release vX.Y.Z" and tags `vX.Y.Z`.
#   7. Pushes the current branch and the tag to `origin`.
#   8. Prints follow-up URLs (GitHub Actions + PyPI).
#
# The CI workflow (`.github/workflows/build-and-publish.yml`) is what actually
# uploads wheels + sdist to PyPI when a `v*` tag is pushed; this script just
# kicks it off cleanly.

set -euo pipefail

if [[ "${1:-}" == "" ]]; then
  echo "Usage: scripts/release.sh <new_version> [-m \"<message>\"]" >&2
  echo "Example: scripts/release.sh 0.3.1 -m \"Add ate_hapc + diagnostics.\"" >&2
  exit 2
fi

NEW_VERSION="$1"
shift || true

MESSAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--message)
      MESSAGE="${2:-}"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2 ;;
  esac
done

# Allow letters for pre-releases (devN / aN / bN / rcN / postN).
if [[ ! "${NEW_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-]?(dev|a|b|rc|post)[0-9]+)?$ ]]; then
  echo "ERROR: '${NEW_VERSION}' is not a valid PEP-440-ish version (X.Y.Z[.devN|aN|bN|rcN])" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYPROJECT="pyproject.toml"
INIT_FILE="python/hapc/__init__.py"

if [[ ! -f "${PYPROJECT}" ]]; then echo "missing ${PYPROJECT}" >&2; exit 1; fi
if [[ ! -f "${INIT_FILE}" ]]; then echo "missing ${INIT_FILE}" >&2; exit 1; fi

# --- 1. Read current version ---------------------------------------------------
CURRENT_PYPROJECT_VERSION="$(grep -E '^version *= *"' "${PYPROJECT}" \
  | head -n1 | sed -E 's/^version *= *"([^"]+)"/\1/')"
CURRENT_INIT_VERSION="$(grep -E '^__version__ *= *"' "${INIT_FILE}" \
  | head -n1 | sed -E 's/^__version__ *= *"([^"]+)"/\1/')"

if [[ -z "${CURRENT_PYPROJECT_VERSION}" || -z "${CURRENT_INIT_VERSION}" ]]; then
  echo "ERROR: could not parse current version from pyproject.toml or __init__.py" >&2
  exit 1
fi

if [[ "${CURRENT_PYPROJECT_VERSION}" != "${CURRENT_INIT_VERSION}" ]]; then
  echo "WARNING: version mismatch before bump: pyproject=${CURRENT_PYPROJECT_VERSION}, __init__=${CURRENT_INIT_VERSION}" >&2
fi

CURRENT_VERSION="${CURRENT_PYPROJECT_VERSION}"
echo ">>> Current version: ${CURRENT_VERSION}"
echo ">>> New version:     ${NEW_VERSION}"

if [[ "${CURRENT_VERSION}" == "${NEW_VERSION}" ]]; then
  echo "ERROR: new version equals current version (${CURRENT_VERSION}); nothing to do." >&2
  exit 1
fi

# Sort + dedupe; if the *smaller* of the two is the new one, refuse.
SMALLER="$(printf '%s\n%s\n' "${CURRENT_VERSION}" "${NEW_VERSION}" | sort -V | head -n1)"
if [[ "${SMALLER}" == "${NEW_VERSION}" ]]; then
  echo "ERROR: new version (${NEW_VERSION}) is older than current (${CURRENT_VERSION})." >&2
  exit 1
fi

# --- 2. Working-tree cleanliness check ----------------------------------------
DIRTY="$(git status --porcelain | grep -Ev '^\?\? ' || true)"
ALLOWED_REGEX='^[ MARC]M? (pyproject\.toml|python/hapc/__init__\.py)$'
UNEXPECTED="$(echo "${DIRTY}" | grep -Ev "${ALLOWED_REGEX}" || true)"
if [[ -n "${UNEXPECTED}" ]]; then
  echo "ERROR: working tree has uncommitted changes outside the two version files:" >&2
  echo "${UNEXPECTED}" >&2
  echo "Commit or stash them before releasing." >&2
  exit 1
fi

# --- 3. Bump versions ----------------------------------------------------------
python3 - "${PYPROJECT}" "${INIT_FILE}" "${CURRENT_VERSION}" "${NEW_VERSION}" <<'PY'
import sys, re, pathlib
pyproject, init, old, new = sys.argv[1:5]
for path in (pyproject, init):
    p = pathlib.Path(path)
    text = p.read_text()
    if path.endswith("pyproject.toml"):
        new_text = re.sub(
            r'^version\s*=\s*"' + re.escape(old) + r'"',
            f'version = "{new}"',
            text,
            count=1,
            flags=re.M,
        )
    else:
        new_text = re.sub(
            r'^__version__\s*=\s*"' + re.escape(old) + r'"',
            f'__version__ = "{new}"',
            text,
            count=1,
            flags=re.M,
        )
    if new_text == text:
        print(f"ERROR: failed to update version in {path} (looking for {old}).", file=sys.stderr)
        sys.exit(1)
    p.write_text(new_text)
    print(f"  updated {path}: {old} -> {new}")
PY

# --- 4. Test ------------------------------------------------------------------
echo ">>> Running pytest (skip R-vs-Python integration)..."
python -m pytest -q --ignore=tests/test_r_vs_python_alpha.py

# --- 5. Commit + tag + push ---------------------------------------------------
COMMIT_MSG="Release v${NEW_VERSION}"
if [[ -n "${MESSAGE}" ]]; then
  COMMIT_MSG="${COMMIT_MSG}

${MESSAGE}"
fi

git add "${PYPROJECT}" "${INIT_FILE}"
git commit -m "${COMMIT_MSG}"
git tag -a "v${NEW_VERSION}" -m "v${NEW_VERSION}"

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo ">>> Pushing ${CURRENT_BRANCH} and tag v${NEW_VERSION}..."
git push origin "${CURRENT_BRANCH}"
git push origin "v${NEW_VERSION}"

cat <<EOF

Released v${NEW_VERSION}.
  GitHub Actions: https://github.com/meixide/hapc/actions
  PyPI:           https://pypi.org/project/hapc/${NEW_VERSION}/

Verify with a fresh install once CI finishes:
  python3 -m venv /tmp/hapc-${NEW_VERSION} && \\
    source /tmp/hapc-${NEW_VERSION}/bin/activate && \\
    pip install --upgrade pip "hapc==${NEW_VERSION}" && \\
    python -c "import hapc; print('hapc', hapc.__version__)"
EOF
