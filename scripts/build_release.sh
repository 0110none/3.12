#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

python -m pip install --upgrade pyinstaller
pyinstaller --noconfirm --clean monitor_alarm.spec

echo "Build finished: dist/安防监控告警系统"
