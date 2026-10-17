#!/usr/bin/env bash
# One git worktree and branch per agent, next to this checkout, so parallel agents never edit the
# same files (docs/32_AGENT_WORKFLOW.md).
#
#   bash scripts/agent_worktree.sh <name> [base]    ../<repo>-<name> on branch agent/<name>
#                                                   (base: the current branch by default)
#   bash scripts/agent_worktree.sh --list
#   bash scripts/agent_worktree.sh --remove <name>  removes the worktree (refuses if it has changes);
#                                                   the branch stays
set -euo pipefail

root=$(git rev-parse --show-toplevel)
repo=$(basename "$root")
usage="usage: agent_worktree.sh <name> [base] | --list | --remove <name>"

case "${1:-}" in
  ""|-h|--help) echo "$usage"; exit 0 ;;
  --list) git -C "$root" worktree list; exit 0 ;;
  --remove)
    name=${2:?"$usage"}
    git -C "$root" worktree remove "$(dirname "$root")/${repo}-${name}"
    echo "removed worktree ${repo}-${name}; branch agent/${name} kept (git branch -d agent/${name} to delete)"
    exit 0 ;;
esac

name=$1
if ! [[ $name =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
  echo "name: lower-case letters, digits and '-' only" >&2; exit 2
fi
base=${2:-$(git -C "$root" rev-parse --abbrev-ref HEAD)}
dir="$(dirname "$root")/${repo}-${name}"
branch="agent/${name}"

if git -C "$root" show-ref --verify --quiet "refs/heads/${branch}"; then
  git -C "$root" worktree add "$dir" "$branch"
else
  git -C "$root" worktree add -b "$branch" "$dir" "$base"
fi

cat <<EOF

worktree: $dir
branch:   $branch (from $base)

next:
  cd "$dir"
  (UI work: cd web && npm ci)        node_modules is not shared between worktrees
  start the agent there; first message: "Read AGENTS.md and docs/31_SESSION_HANDOFF.md, then ..."
  push: git push -u origin $branch   the integrator merges it after the checks pass
EOF
