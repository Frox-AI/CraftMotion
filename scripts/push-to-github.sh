#!/usr/bin/env bash
# Create a GitHub repo from this folder, push it, and wait for the Windows build.
#
#   bash scripts/push-to-github.sh [repo-name] [--public]
#
# Needs the GitHub CLI (https://cli.github.com) and `gh auth login` done once.
# Everything here is idempotent: re-running just commits and pushes again.
set -euo pipefail

REPO="${1:-craftmotion}"
VIS="--private"
[[ "${2:-}" == "--public" ]] && VIS="--public"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "· repo root: $PWD"

command -v git >/dev/null || { echo "git is required"; exit 1; }

# ---------------------------------------------------------------- git identity
if ! git config user.email >/dev/null; then
  git config user.name  "$(git config --global user.name  || echo CraftMotion)"
  git config user.email "$(git config --global user.email || echo craftmotion@example.com)"
fi

# ------------------------------------------------------------------- init/commit
if [ ! -d .git ]; then
  git init -b main
fi
git add -A
if ! git diff --cached --quiet; then
  git commit -m "CraftMotion 1.1.0 + SIGNAL / NOISE edit"
else
  echo "· nothing new to commit"
fi

# ------------------------------------------------------------------ create/push
if command -v gh >/dev/null; then
  if ! gh repo view "$REPO" >/dev/null 2>&1; then
    gh repo create "$REPO" "$VIS" --source . --remote origin --push
  else
    git remote get-url origin >/dev/null 2>&1 || \
      git remote add origin "$(gh repo view "$REPO" --json sshUrl -q .sshUrl)"
    git push -u origin main
  fi
  OWNER="$(gh repo view "$REPO" --json owner -q .owner.login)"
  echo
  echo "✓ pushed → https://github.com/$OWNER/$REPO"
  echo "· starting the Windows build…"

  PUSHED=$(gh workflow list --limit 20 --json name,id -q '.[] | select(.name|test("Build Windows")) | .id' | head -1)
  if [ -n "${PUSHED:-}" ]; then
    gh workflow run "$PUSHED" >/dev/null 2>&1 || true
    sleep 8
    gh run watch "$(gh run list --workflow "$PUSHED" --limit 1 --json databaseId -q '.[0].databaseId')" || true
    OUT="${REPO}-artifact"
    mkdir -p "$OUT"
    gh run download --name CraftMotion-Windows -D "$OUT" || true
    [ -f "$OUT/CraftMotion.exe" ] && echo "✓ CraftMotion.exe → $OUT/CraftMotion.exe"
    gh run download --name CraftMotion-MSI -D "$OUT" 2>/dev/null || true
    ls -la "$OUT" 2>/dev/null || true
  else
    echo "! workflow not found yet — push again, then check the Actions tab"
  fi
else
  echo
  echo "gh CLI not found. Create an empty repo at https://github.com/new, then:"
  echo "  git remote add origin https://github.com/<you>/$REPO.git"
  echo "  git push -u origin main"
fi
