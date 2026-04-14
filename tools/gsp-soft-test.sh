#!/usr/bin/env bash
# gsp-soft-test.sh — reversible, non-destructive GSP validation run.
#
# What this does:
#   1. Captures pre-state (nvidia bound, vLLM listening, dmesg baseline).
#   2. Asks the operator to confirm — then takes down vLLM and unbinds nvidia.
#   3. Runs ONLY read-only probes (probe_bar0, probe_gsp_falcon, probe_ptimer).
#      Does NOT run gsp_boot end-to-end. Does NOT exercise falcon_reset,
#      fw_load, bcr_start, FSP, poll_lockdown, or rpc_channel — all of
#      which write to GPU registers and can wedge the device.
#   4. Logs every probe's output (stdout + dmesg delta) to a dated file.
#   5. Rebinds nvidia, restarts vLLM (host-specific — see RESTART_VLLM_CMD).
#   6. Confirms post-state matches pre-state.
#
# What this does NOT do:
#   - Run gsp_boot (Step 3+ is destructive)
#   - Attempt FSP (Step 7 still prints TODO in boot.s:123)
#   - Test any part of the bring-up chain that requires successful FSP
#
# Failure modes:
#   - If unbind fails, we abort before any probe runs (GPU still owned by nvidia).
#   - If a probe SIGBUSes, it crashes itself; GPU state is unchanged.
#   - If rebind fails at the end, `nvidia-smi` will fail → reboot required.
#     This script logs the exact rebind command so an operator can retry by hand.
#
# Usage:
#   ./tools/gsp-soft-test.sh [--dry-run] [--keep-vllm-down]
#
# Exit codes:
#   0  — all probes passed, state restored cleanly
#   1  — pre-flight failed (tools missing, permissions, etc.)
#   2  — unbind failed (bailed before touching probes)
#   3  — at least one probe failed (but state restored)
#   4  — rebind failed (GPU in unbound state — MANUAL RECOVERY REQUIRED)

set -euo pipefail

DRY_RUN=0
KEEP_VLLM_DOWN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --keep-vllm-down) KEEP_VLLM_DOWN=1 ;;
        --help|-h)
            sed -n '1,/^$/p' "$0" | sed 's|^# \{0,1\}||'
            exit 0 ;;
        *) echo "unknown arg: $arg"; exit 1 ;;
    esac
done

LITHOS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GSP_DIR="$LITHOS_ROOT/GSP"
GPU_SETUP="$LITHOS_ROOT/tools/gpu-setup.sh"
LOG_DIR="$LITHOS_ROOT/logs/gsp-soft-test"
mkdir -p "$LOG_DIR"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOG_DIR/$TS.log"
DMESG_BEFORE="$LOG_DIR/$TS.dmesg-before.txt"
DMESG_AFTER="$LOG_DIR/$TS.dmesg-after.txt"

# vLLM restart command — host-specific. If the operator wants a different
# restart sequence, they can override via env var.
RESTART_VLLM_CMD="${RESTART_VLLM_CMD:-nohup /home/ubuntu/launch_vllm.sh > /home/ubuntu/vllm.log 2>&1 &}"

# --- output helpers ---------------------------------------------------------
is_tty() { [ -t 1 ]; }
color() { if is_tty; then printf '\033[%sm%s\033[0m' "$1" "$2"; else printf '%s' "$2"; fi; }
hdr()   { echo; color "1;36" "== $* =="; echo; } >&2
ok()    { color "1;32" "✓"; echo " $*"; } >&2
warn()  { color "1;33" "⚠"; echo " $*"; } >&2
err()   { color "1;31" "✗"; echo " $*"; } >&2
run()   {
    if [ "$DRY_RUN" = 1 ]; then
        color "1;33" "DRY"; echo " $*"; return 0
    fi
    echo "+ $*" >> "$LOG"
    "$@" 2>&1 | tee -a "$LOG"
}

# --- 1. pre-flight ----------------------------------------------------------
hdr "pre-flight"
{
    echo "timestamp: $TS"
    echo "host: $(hostname)"
    echo "user: $(whoami)"
    echo "lithos_root: $LITHOS_ROOT"
} | tee -a "$LOG"

[ -x "$GPU_SETUP" ]        || { err "missing tools/gpu-setup.sh"; exit 1; }
[ -x "$GSP_DIR/probe_bar0" ]       || { err "missing GSP/probe_bar0 binary — build it first"; exit 1; }
[ -x "$GSP_DIR/probe_gsp_falcon" ] || { err "missing GSP/probe_gsp_falcon binary"; exit 1; }
# probe_ptimer is nice-to-have; don't fail if absent
[ -x "$GSP_DIR/probe_ptimer" ]     || warn "probe_ptimer binary missing (optional)"

if command -v nvidia-smi >/dev/null; then
    run nvidia-smi --query-gpu=name,driver_version,pstate,persistence_mode --format=csv
else
    warn "nvidia-smi not on PATH — that's fine, we'll read sysfs directly"
fi

# capture dmesg baseline (sudo may be needed for full kernel ring)
if sudo -n true 2>/dev/null; then
    sudo dmesg --time-format iso > "$DMESG_BEFORE" 2>/dev/null || dmesg --time-format iso > "$DMESG_BEFORE"
else
    dmesg --time-format iso > "$DMESG_BEFORE" 2>/dev/null || warn "cannot read dmesg (ok if no root)"
fi

# is vLLM up?
if ss -tlnp 2>/dev/null | grep -q ':8001 '; then
    VLLM_WAS_UP=1
    ok "vLLM listening on :8001 — will stop, then restart at end unless --keep-vllm-down"
else
    VLLM_WAS_UP=0
    warn "vLLM not listening on :8001 — skip-restart by default"
fi

# --- 2. confirm ------------------------------------------------------------
hdr "plan"
cat <<EOF | tee -a "$LOG"
About to:
  1. Stop vLLM (will restart at end)       [vllm_was_up=$VLLM_WAS_UP]
  2. Unbind nvidia via tools/gpu-setup.sh  [requires sudo]
  3. Run read-only probes:
       - GSP/probe_bar0         (reads PMC_BOOT_0, a few scratch regs)
       - GSP/probe_gsp_falcon   (reads FALCON_HWCFG2, mailbox, OS state)
       - GSP/probe_ptimer       (reads PTIMER, 10 samples)       [if present]
  4. Rebind nvidia
  5. Restart vLLM (unless --keep-vllm-down)
  6. Compare pre/post dmesg for anomalies

NO writes to any GPU register. NO falcon_reset. NO FSP.
Dry-run mode: $DRY_RUN

EOF

if [ "$DRY_RUN" = 0 ]; then
    read -rp "Proceed? Type YES to continue: " answer
    [ "$answer" = "YES" ] || { echo "aborted."; exit 1; }
fi

# --- 3. stop vLLM ---------------------------------------------------------
if [ "$VLLM_WAS_UP" = 1 ]; then
    hdr "stopping vLLM"
    run pkill -f "vllm.entrypoints" || warn "no vllm.entrypoints processes to kill"
    sleep 3
    if ss -tlnp 2>/dev/null | grep -q ':8001 '; then
        err ":8001 still listening — something else is holding it. Aborting."
        exit 2
    fi
    ok "vLLM stopped"
fi

# --- 4. unbind nvidia ------------------------------------------------------
hdr "unbind nvidia"
if [ "$DRY_RUN" = 1 ]; then
    run sudo "$GPU_SETUP" unbind --dry-run
else
    if ! sudo "$GPU_SETUP" unbind --yes; then
        err "gpu-setup.sh unbind failed — GPU still owned by nvidia, no probes ran"
        # Try to restart vLLM since nvidia is still up
        if [ "$VLLM_WAS_UP" = 1 ] && [ "$KEEP_VLLM_DOWN" = 0 ]; then
            warn "restarting vLLM since nothing changed"
            bash -c "$RESTART_VLLM_CMD"
        fi
        exit 2
    fi
fi
ok "nvidia unbound"

# --- 5. run probes --------------------------------------------------------
PROBE_FAIL=0
hdr "probe_bar0 (PMC_BOOT_0, a few registers)"
if [ "$DRY_RUN" = 1 ]; then
    run echo sudo "$GSP_DIR/probe_bar0"
else
    sudo "$GSP_DIR/probe_bar0" 2>&1 | tee -a "$LOG" || { err "probe_bar0 failed"; PROBE_FAIL=1; }
fi

hdr "probe_gsp_falcon (HWCFG2, mailbox, OS state)"
if [ "$DRY_RUN" = 1 ]; then
    run echo sudo "$GSP_DIR/probe_gsp_falcon"
else
    sudo "$GSP_DIR/probe_gsp_falcon" 2>&1 | tee -a "$LOG" || { err "probe_gsp_falcon failed"; PROBE_FAIL=1; }
fi

if [ -x "$GSP_DIR/probe_ptimer" ]; then
    hdr "probe_ptimer (10 timestamp samples)"
    if [ "$DRY_RUN" = 1 ]; then
        run echo sudo "$GSP_DIR/probe_ptimer"
    else
        sudo "$GSP_DIR/probe_ptimer" 2>&1 | tee -a "$LOG" || { err "probe_ptimer failed"; PROBE_FAIL=1; }
    fi
fi

# --- 6. rebind nvidia -----------------------------------------------------
hdr "rebind nvidia"
REBIND_OK=0
if [ "$DRY_RUN" = 1 ]; then
    run sudo "$GPU_SETUP" rebind --dry-run
    REBIND_OK=1
else
    if sudo "$GPU_SETUP" rebind; then
        REBIND_OK=1
        ok "nvidia rebound"
    else
        err "REBIND FAILED — GPU currently UNBOUND."
        err "Manual recovery: sudo $GPU_SETUP rebind (or reboot)"
        echo "Keeping log at $LOG" >&2
        exit 4
    fi
fi

# --- 7. restart vLLM ------------------------------------------------------
if [ "$VLLM_WAS_UP" = 1 ] && [ "$KEEP_VLLM_DOWN" = 0 ] && [ "$DRY_RUN" = 0 ]; then
    hdr "restart vLLM"
    bash -c "$RESTART_VLLM_CMD"
    sleep 3
    warn "vLLM starting — CUDA graph JIT takes ~3–5 min; check /home/ubuntu/vllm.log"
fi

# --- 8. post-check --------------------------------------------------------
hdr "post-check"
if command -v nvidia-smi >/dev/null && [ "$DRY_RUN" = 0 ]; then
    if nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
        ok "nvidia-smi healthy"
    else
        err "nvidia-smi returned an error after rebind"
        exit 4
    fi
fi

if sudo -n true 2>/dev/null && [ "$DRY_RUN" = 0 ]; then
    sudo dmesg --time-format iso > "$DMESG_AFTER" 2>/dev/null
    NEW_LINES=$(diff "$DMESG_BEFORE" "$DMESG_AFTER" 2>/dev/null | grep '^>' | wc -l || echo 0)
    if [ "$NEW_LINES" -gt 0 ]; then
        warn "$NEW_LINES new dmesg lines during the test — review $DMESG_AFTER"
        diff "$DMESG_BEFORE" "$DMESG_AFTER" | grep '^>' | head -20 | tee -a "$LOG"
    else
        ok "no new dmesg lines"
    fi
fi

hdr "summary"
echo "log: $LOG"
if [ "$PROBE_FAIL" = 1 ]; then
    err "one or more probes failed — state restored, check log"
    exit 3
else
    ok "all probes passed, state restored"
    exit 0
fi
