#!/bin/zsh
# sync.sh — Push all changes to GitHub with a single command
# Usage: ./sync.sh "your commit message"
#        ./sync.sh  (uses a default timestamped message)

cd "$(dirname "$0")"

MSG="${1:-"auto-sync: $(date '+%Y-%m-%d %H:%M')"}"

git add -A
git commit -m "$MSG"
git push origin main
