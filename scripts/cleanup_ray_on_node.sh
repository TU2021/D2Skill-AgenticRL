#!/usr/bin/env bash
# Run this on the target node to clean up stale Ray processes
# (example: srun -p gpu -w hd02-gpu1-0033 bash scripts/cleanup_ray_on_node.sh)
set -e
echo "=== Current node: $(hostname) ==="
echo "Searching for Ray-related processes..."
ROWS=$(ps -u "$USER" -o pid,cmd --no-headers 2>/dev/null | grep -E 'ray|raylet|gcs_server|dashboard/agent|plasma_store' | grep -v grep || true)
if [[ -z "$ROWS" ]]; then
  echo "No Ray-related processes found."
  exit 0
fi
echo "$ROWS"
PIDS=$(echo "$ROWS" | awk '{print $1}')
echo "About to run kill -9 on: $PIDS"
for pid in $PIDS; do
  kill -9 "$pid" 2>/dev/null && echo "Terminated PID $pid" || echo "Failed to terminate $pid (possibly already exited)"
done
echo "Cleanup complete."
