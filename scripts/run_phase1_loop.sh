#!/bin/bash
# Wrapper that restarts Python between prompt batches to defeat the
# transformers attention-tensor leak. The Python script processes a few
# prompts, then exits with code 2 when GPU memory exceeds the threshold;
# this loop relaunches and the resume logic picks up where it left off.
# Exit code 0 means all targeted prompts are done.

set -u
cd "$(dirname "$0")/.."   # repo root
LOG="${LOG:-phase1_full.log}"
MAX_RESTARTS="${MAX_RESTARTS:-150}"

EXIT_DONE=0
EXIT_RESTART=2

restarts=0
while [ "$restarts" -lt "$MAX_RESTARTS" ]; do
    restarts=$((restarts+1))
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') — invocation $restarts ===" >> "$LOG"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python -u scripts/phase1_baseline.py >> "$LOG" 2>&1
    code=$?
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') — exited with code $code ===" >> "$LOG"

    if [ "$code" -eq "$EXIT_DONE" ]; then
        echo "=== $(date '+%Y-%m-%d %H:%M:%S') — all prompts done ===" >> "$LOG"
        exit 0
    elif [ "$code" -eq "$EXIT_RESTART" ]; then
        sleep 5
    else
        # Unexpected error — could be OOM that bypassed our threshold,
        # or a transient. Retry; resume logic protects completed work.
        echo "=== $(date '+%Y-%m-%d %H:%M:%S') — unexpected exit, retrying in 30s ===" >> "$LOG"
        sleep 30
    fi
done

echo "=== $(date '+%Y-%m-%d %H:%M:%S') — hit $MAX_RESTARTS restart limit, giving up ===" >> "$LOG"
exit 1
