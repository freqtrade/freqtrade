#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SOURCE_ROOT="$ROOT/tools/strategy_research_agent/skills"
TARGET_ROOT="${CODEX_AGENT_SKILLS_DIR:-$HOME/.agents/skills}"

if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "Missing source skill directory: $SOURCE_ROOT" >&2
  exit 1
fi

mkdir -p "$TARGET_ROOT"

for skill_dir in "$SOURCE_ROOT"/*; do
  [[ -d "$skill_dir" ]] || continue
  skill_name="$(basename "$skill_dir")"
  target="$TARGET_ROOT/$skill_name"
  rm -rf "$target"
  mkdir -p "$(dirname "$target")"
  cp -R "$skill_dir" "$target"
  echo "Installed skill: $skill_name -> $target"
done

echo "Strategy research skills installed into $TARGET_ROOT"
