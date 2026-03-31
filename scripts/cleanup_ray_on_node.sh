#!/usr/bin/env bash
# 在目标节点上执行以清理 Ray 残留进程（例如：srun -p gpu -w hd02-gpu1-0033 bash scripts/cleanup_ray_on_node.sh）
set -e
echo "=== 当前节点: $(hostname) ==="
echo "查找 Ray 相关进程..."
ROWS=$(ps -u "$USER" -o pid,cmd --no-headers 2>/dev/null | grep -E 'ray|raylet|gcs_server|dashboard/agent|plasma_store' | grep -v grep || true)
if [[ -z "$ROWS" ]]; then
  echo "未发现 Ray 相关进程。"
  exit 0
fi
echo "$ROWS"
PIDS=$(echo "$ROWS" | awk '{print $1}')
echo "即将 kill -9: $PIDS"
for pid in $PIDS; do
  kill -9 "$pid" 2>/dev/null && echo "已结束 PID $pid" || echo "结束 $pid 失败（可能已退出）"
done
echo "清理完成。"
