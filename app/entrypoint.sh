#!/bin/bash
# Batch container entrypoint.
# 1. (venue ibkr) wait for the ib-gateway sidecar to accept connections on 4002
# 2. (TACTIS_CHECKPOINT_S3 set) download checkpoint + norm_stats from S3, run
#    fresh-data fetch + model inference to emit today's weights JSON
# 3. run the day-loop with LiveGuard dry-run defaults; --dry-run can be
#    disabled ONLY by job env (TACTIS_LIVE=1 + venue live:true in application.yaml)
set -euo pipefail

VENT_TRADE_DIR=${TACTIS_REPO_DIR:-/opt}
cd "$VENT_TRADE_DIR/app"

GATEWAY_HOST=${IB_GATEWAY_HOST:-127.0.0.1}
GATEWAY_PORT=${GATEWAY_PORT:-4002}

if [ "${TACTIS_VENUE:-}" = "ibkr" ]; then
  echo "waiting for IB Gateway on ${GATEWAY_HOST}:${GATEWAY_PORT}..."
  for i in $(seq 1 60); do
    if timeout 2 bash -c "</dev/tcp/${GATEWAY_HOST}/${GATEWAY_PORT}" 2>/dev/null; then
      echo "gateway up"; break
    fi
    sleep 5
  done
fi

WEIGHTS_FILE="${TACTIS_WEIGHTS_FILE:-}"

if [ -n "${TACTIS_CHECKPOINT_S3:-}" ]; then
  echo "=== inference path: checkpoint ${TACTIS_CHECKPOINT_S3} ==="
  MODEL_DIR=/opt/model
  DATA_DIR=/opt/data
  mkdir -p "$MODEL_DIR" "$DATA_DIR"

  aws s3 cp "${TACTIS_CHECKPOINT_S3}" "$MODEL_DIR/checkpoint.pth"
  aws s3 cp "${TACTIS_NORM_STATS_S3:-s3://<AWS_ACCOUNT_ID>-tactis-build/checkpoints/norm_stats.json}" \
    "$MODEL_DIR/norm_stats.json"
  echo "[entrypoint] checkpoint + norm_stats downloaded"

  # fresh price data: primary (yfinance/baostock); fallback (IBKR bars) only
  # makes sense for the ibkr venue where the gateway is already up
  DATA_CSV="$DATA_DIR/all_stocks.csv"
  if python3 data_fetch.py --out "$DATA_CSV" \
        --tickers-csv "${TACTIS_UNIVERSE_CSV:-/opt/app/universe/us_all.csv}" \
        --cn-tickers-csv "${TACTIS_UNIVERSE_CSV:-/opt/app/universe/cn_all.csv}" \
        --days "${TACTIS_DATA_DAYS:-40}"; then
    echo "[entrypoint] primary data fetch OK"
  elif [ "${TACTIS_VENUE:-}" = "ibkr" ]; then
    echo "[entrypoint] primary fetch failed — falling back to IBKR historical bars"
    python3 data_fetch.py --via ibkr --out "$DATA_CSV" --update \
      --tickers-csv "${TACTIS_UNIVERSE_CSV:-/opt/app/universe/us_all.csv}" \
      --days "${TACTIS_DATA_DAYS:-40}"
  else
    echo "FATAL: data fetch failed and no fallback for venue ${TACTIS_VENUE}" >&2
    exit 3
  fi

  WEIGHTS_FILE=/opt/app/weights_inferred.json
  echo "[entrypoint] running inference -> $WEIGHTS_FILE"
  python3 infer.py \
    --checkpoint "$MODEL_DIR/checkpoint.pth" \
    --data "$DATA_CSV" \
    --norm-stats "$MODEL_DIR/norm_stats.json" \
    --out "$WEIGHTS_FILE" \
    --num-samples "${TACTIS_NUM_SAMPLES:-64}" \
    --max-assets "${TACTIS_MAX_ASSETS:-20}" \
    ${TACTIS_DEVICE:+--device "$TACTIS_DEVICE"}
  echo "=== inference done: $WEIGHTS_FILE ==="
fi

python3 run_live.py \
  --venue "${TACTIS_VENUE:?must set TACTIS_VENUE}" \
  --weights "${WEIGHTS_FILE:?must set TACTIS_WEIGHTS_FILE or TACTIS_CHECKPOINT_S3}" \
  $( [ "${TACTIS_FORCE_DRYRUN:-1}" = "1" ] && echo --dry-run )
