#!/usr/bin/env sh
# Chạy M31 trên server Linux (cần kích hoạt .venv trước, xem README "Triển khai lên server").
# Đổi cổng/địa chỉ bằng biến môi trường:  PORT=8080 ./run_server.sh   |   ADDRESS=127.0.0.1 ./run_server.sh
set -e
cd "$(dirname "$0")"
PORT="${PORT:-8501}"
ADDRESS="${ADDRESS:-0.0.0.0}"
exec python -m streamlit run app.py \
  --server.address "$ADDRESS" \
  --server.port "$PORT" \
  --server.headless true \
  --browser.gatherUsageStats false
