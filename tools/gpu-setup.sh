#!/usr/bin/env bash
# gpu-setup.sh — transition a GH200 host between "nvidia-bound + vLLM running"
# and "GPU available to lithos userspace code".
#
# Called for in docs/runtime/gsp-native.md:508 ("Add gpu-setup.sh script that
# unbinds nvidia.ko and sets permissions") and consumed by the bring-up flow
# described in GSP/fsp_plan.md (lithos launcher expects BAR0/BAR4 mmap to
# succeed from userspace; nvidia.ko holds exclusive PCI binding while loaded,
# so either (a) we unbind & bind vfio-pci, or (b) we keep nvidia and chmod the
# sysfs resource files for read-only probing).
#
# Subcommands:
#   status   — report current state, read-only
#   unbind   — tear down nvidia, bind vfio-pci  (destructive, guarded)
#   rebind   — tear down vfio-pci, restore nvidia (destructive, guarded)
#   chmod    — widen sysfs resource0/resource1 perms, keep nvidia bound (race-y)
#   doctor   — verbose diagnostic dump, no state changes
#
# Flags:
#   --dry-run     print the commands that would run, do not execute
#   --yes         skip interactive confirmation for destructive actions
#   --pci BDF     override detected PCI BDF (default: auto-detect, falls back
#                 to dd:00.0 which is the known GH200 slot on this host)
#   --group NAME  group for `chmod` mode (default: current user's primary group)
#
# Safety contract:
#   * set -euo pipefail
#   * Never rmmod while /dev/nvidia* is open (checked via fuser/lsof).
#   * Bail on unbind if vLLM is bound to :8001 (ss -tlnp).
#   * Every modifying action is dry-runnable.
#   * Colored output on tty; plain otherwise.
#
# Known limits: see bottom of file.

set -euo pipefail

# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

if [[ -t 1 ]]; then
    C_RED=$'\033[31m'; C_GRN=$'\033[32m'; C_YEL=$'\033[33m'
    C_BLU=$'\033[34m'; C_DIM=$'\033[2m';  C_RST=$'\033[0m'
else
    C_RED=''; C_GRN=''; C_YEL=''; C_BLU=''; C_DIM=''; C_RST=''
fi

log()   { printf '%s[gpu-setup]%s %s\n' "$C_BLU" "$C_RST" "$*"; }
ok()    { printf '%s[ ok ]%s %s\n'      "$C_GRN" "$C_RST" "$*"; }
warn()  { printf '%s[warn]%s %s\n'      "$C_YEL" "$C_RST" "$*" >&2; }
err()   { printf '%s[err ]%s %s\n'      "$C_RED" "$C_RST" "$*" >&2; }
die()   { err "$*"; exit 1; }

DRY_RUN=0
ASSUME_YES=0
PCI_BDF=""
TARGET_GROUP=""

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

SUBCMD=""
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run) DRY_RUN=1; shift;;
            --yes|-y)  ASSUME_YES=1; shift;;
            --pci)     PCI_BDF="${2:-}"; shift 2;;
            --group)   TARGET_GROUP="${2:-}"; shift 2;;
            -h|--help) usage; exit 0;;
            status|unbind|rebind|chmod|doctor)
                [[ -z "$SUBCMD" ]] || die "more than one subcommand given"
                SUBCMD="$1"; shift;;
            *) die "unknown argument: $1 (try --help)";;
        esac
    done
    [[ -n "$SUBCMD" ]] || { usage; exit 2; }
}

usage() {
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
}

# run <cmd...>: honors --dry-run.  Use for any state-mutating call.
run() {
    if (( DRY_RUN )); then
        printf '%s[dry ]%s %s\n' "$C_DIM" "$C_RST" "$*"
    else
        log "exec: $*"
        "$@"
    fi
}

# run_sh "shell string": honors --dry-run, used when redirection / tee is required.
run_sh() {
    local s="$1"
    if (( DRY_RUN )); then
        printf '%s[dry ]%s sh -c %q\n' "$C_DIM" "$C_RST" "$s"
    else
        log "exec-sh: $s"
        bash -c "$s"
    fi
}

confirm() {
    local prompt="$1"
    if (( ASSUME_YES )); then return 0; fi
    if [[ ! -t 0 ]]; then
        die "refusing to do destructive action non-interactively without --yes"
    fi
    read -r -p "$prompt [type YES to proceed]: " ans
    [[ "$ans" == "YES" ]] || die "cancelled"
}

# ---------------------------------------------------------------------------
# Discovery (verify-before-write: no hardcoded register offsets or modules)
# ---------------------------------------------------------------------------

# detect_pci_bdf: pick the first NVIDIA 3D/VGA/display controller.
# Falls back to the user-supplied --pci or the documented dd:00.0 slot.
detect_pci_bdf() {
    if [[ -n "$PCI_BDF" ]]; then echo "$PCI_BDF"; return; fi
    local candidates
    candidates=$(lspci -D -nn 2>/dev/null \
        | awk '/\[10de:/ && /(3D|VGA|Display)/ {print $1}' \
        || true)
    if [[ -z "$candidates" ]]; then
        warn "lspci found no NVIDIA GPU; falling back to 0000:dd:00.0"
        echo "0000:dd:00.0"; return
    fi
    # Take the first candidate
    echo "$candidates" | head -n1
}

# list_loaded_nvidia_modules: print loaded nvidia* modules in *unload order*
# (dependents first). Uses lsmod, not a hardcoded list.
list_loaded_nvidia_modules() {
    # lsmod columns: Module Size UsedBy
    # We want leaves (nothing depending on them) first. Do a topo sort
    # by repeatedly picking modules with UsedBy count 0 among our set.
    local set
    set=$(lsmod | awk 'NR>1 && $1 ~ /^(nvidia|nvidia_)/ {print $1}')
    [[ -n "$set" ]] || return 0

    local remaining="$set"
    local order=()
    local guard=0
    while [[ -n "$remaining" ]]; do
        local progressed=0
        local new_remaining=""
        for mod in $remaining; do
            # Count how many *currently-remaining* modules depend on $mod.
            local deps
            deps=$(lsmod | awk -v m="$mod" 'NR>1 && $1==m {print $4}')
            local depcount=0
            if [[ -n "$deps" && "$deps" != "-" ]]; then
                IFS=',' read -r -a darr <<< "$deps"
                for d in "${darr[@]}"; do
                    d="${d// /}"
                    [[ -z "$d" ]] && continue
                    # only count deps still in remaining
                    if echo "$remaining" | grep -qw -- "$d"; then
                        depcount=$((depcount+1))
                    fi
                done
            fi
            if (( depcount == 0 )); then
                order+=("$mod")
                progressed=1
            else
                new_remaining+="$mod "
            fi
        done
        remaining="${new_remaining% }"
        if (( ! progressed )); then
            warn "module dep cycle (shouldn't happen); punting remainder: $remaining"
            for m in $remaining; do order+=("$m"); done
            break
        fi
        guard=$((guard+1))
        (( guard < 20 )) || { warn "topo sort guard hit"; break; }
    done
    printf '%s\n' "${order[@]}"
}

# current_driver: which driver is currently bound to $1 (BDF)? Empty if none.
current_driver() {
    local bdf="$1"
    if [[ -L "/sys/bus/pci/devices/$bdf/driver" ]]; then
        basename "$(readlink -f "/sys/bus/pci/devices/$bdf/driver")"
    fi
}

# gpu_ids: print "VENDOR DEVICE" for a BDF (e.g. "10de 2342").
gpu_ids() {
    local bdf="$1"
    local v d
    v=$(cat "/sys/bus/pci/devices/$bdf/vendor" 2>/dev/null | sed 's/^0x//')
    d=$(cat "/sys/bus/pci/devices/$bdf/device" 2>/dev/null | sed 's/^0x//')
    echo "$v $d"
}

# vllm_on_8001: 0 if something is LISTENing on :8001, 1 otherwise.
# Uses `ss`; we grep for 8001 in a listen context.
vllm_on_8001() {
    if command -v ss >/dev/null 2>&1; then
        ss -tlnp 2>/dev/null | awk '$4 ~ /:8001$/ {found=1} END{exit !found}'
    else
        # Fallback: /proc/net/tcp — 8001 = 0x1F41
        awk '$2 ~ /:1F41$/ && $4=="0A" {found=1} END{exit !found}' /proc/net/tcp 2>/dev/null
    fi
}

# nvidia_dev_open: 0 if anything has /dev/nvidia* open, 1 otherwise.
nvidia_dev_open() {
    local any=1
    shopt -s nullglob
    local devs=(/dev/nvidia*)
    shopt -u nullglob
    (( ${#devs[@]} )) || return 1
    if command -v fuser >/dev/null 2>&1; then
        if fuser -s "${devs[@]}" 2>/dev/null; then any=0; fi
    elif command -v lsof >/dev/null 2>&1; then
        if lsof "${devs[@]}" >/dev/null 2>&1; then any=0; fi
    else
        warn "neither fuser nor lsof installed; cannot check /dev/nvidia* openers"
        # Conservative: assume busy.
        any=0
    fi
    return "$any"
}

vfio_pci_available() {
    [[ -d /sys/bus/pci/drivers/vfio-pci ]]
}

# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

cmd_status() {
    local bdf ids drv
    bdf=$(detect_pci_bdf)
    read -r ven dev <<< "$(gpu_ids "$bdf")"
    drv=$(current_driver "$bdf")

    echo "PCI BDF      : $bdf"
    echo "Vendor:Device: ${ven:-?}:${dev:-?}"
    echo "Bound driver : ${drv:-<none>}"

    if vfio_pci_available; then
        ok  "vfio-pci driver is registered (/sys/bus/pci/drivers/vfio-pci)"
    else
        warn "vfio-pci driver is NOT registered — modprobe vfio_pci needed before unbind"
    fi

    echo -n "Loaded nvidia modules : "
    local mods
    mods=$(lsmod | awk 'NR>1 && $1 ~ /^(nvidia|nvidia_)/ {print $1}' | tr '\n' ' ')
    echo "${mods:-<none>}"

    if vllm_on_8001; then
        warn "something is LISTENing on :8001 (assumed vLLM)"
    else
        ok "nothing LISTENing on :8001"
    fi

    if [[ -e "/sys/bus/pci/devices/$bdf/resource0" ]]; then
        local info
        info=$(stat -c 'mode=%a user=%U group=%G' "/sys/bus/pci/devices/$bdf/resource0")
        echo "resource0    : $info"
    else
        warn "no resource0 under $bdf (unusual)"
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
            ok "nvidia-smi responsive"
        else
            warn "nvidia-smi installed but not responsive (expected if unbound)"
        fi
    fi
}

# ---------------------------------------------------------------------------
# unbind — nvidia -> vfio-pci
# ---------------------------------------------------------------------------

cmd_unbind() {
    local bdf drv
    bdf=$(detect_pci_bdf)
    drv=$(current_driver "$bdf")

    if [[ "$drv" == "vfio-pci" ]]; then
        ok "already bound to vfio-pci; nothing to do"
        return 0
    fi
    if [[ "$drv" != "nvidia" && -n "$drv" ]]; then
        die "unexpected driver bound: $drv (expected nvidia or none)"
    fi

    # Guard 1: vLLM on :8001
    if vllm_on_8001; then
        die "vLLM (or something) is LISTENing on :8001. Stop it first."
    fi

    # Guard 2: /dev/nvidia* openers
    if nvidia_dev_open; then
        die "/dev/nvidia* is open by some process. Close CUDA/Triton consumers first."
    fi

    confirm "About to rmmod nvidia stack at $bdf and bind vfio-pci. Sure?"

    # Best-effort: stop persistence daemon if systemd-managed.
    if command -v systemctl >/dev/null 2>&1 \
            && systemctl is-active --quiet nvidia-persistenced 2>/dev/null; then
        run systemctl stop nvidia-persistenced
    else
        log "nvidia-persistenced not systemd-active; skipping"
    fi

    # Kill any `nvidia-smi -pm 1` or similar straggler: only warn, don't kill.
    if pgrep -a nvidia-persistenced >/dev/null 2>&1; then
        warn "nvidia-persistenced still running (not via systemd). You may need 'pkill nvidia-persistenced'."
    fi

    # Unload in dependent-first order, discovered from live lsmod.
    local mods
    mapfile -t mods < <(list_loaded_nvidia_modules)
    if (( ${#mods[@]} == 0 )); then
        warn "no nvidia* modules loaded (already unloaded?)"
    else
        log "unload order: ${mods[*]}"
        for m in "${mods[@]}"; do
            run rmmod "$m"
        done
    fi

    # Ensure vfio-pci module is loaded.
    if ! vfio_pci_available; then
        log "loading vfio_pci"
        run modprobe vfio_pci
        vfio_pci_available || warn "vfio-pci still not present after modprobe; the /new_id bind will fail"
    fi

    # If the device is still bound to *anything*, unbind first.
    if [[ -e "/sys/bus/pci/devices/$bdf/driver" ]]; then
        log "unbinding $bdf from $(current_driver "$bdf")"
        run_sh "echo '$bdf' > /sys/bus/pci/devices/$bdf/driver/unbind"
    fi

    # Set driver_override, then bind via new_id.
    # (new_id takes "vendor device" in hex with no 0x.)
    local ven dev
    read -r ven dev <<< "$(gpu_ids "$bdf")"
    [[ -n "$ven" && -n "$dev" ]] || die "could not read vendor/device from sysfs"

    run_sh "echo vfio-pci > /sys/bus/pci/devices/$bdf/driver_override"
    run_sh "echo '$ven $dev' > /sys/bus/pci/drivers/vfio-pci/new_id"
    # new_id triggers an automatic bind of all matching unbound devices.
    # If driver_override already routed it, the bind happened; otherwise
    # explicit bind:
    if [[ "$(current_driver "$bdf" || true)" != "vfio-pci" ]]; then
        run_sh "echo '$bdf' > /sys/bus/pci/drivers/vfio-pci/bind" || true
    fi

    # Verify.
    local now
    now=$(current_driver "$bdf")
    if [[ "$now" == "vfio-pci" ]]; then
        ok "bound to vfio-pci"
        run_sh "lspci -k -s '${bdf#0000:}' | sed 's/^/    /'"
    else
        die "bind failed: current driver is '${now:-<none>}'"
    fi
}

# ---------------------------------------------------------------------------
# rebind — vfio-pci -> nvidia
# ---------------------------------------------------------------------------

cmd_rebind() {
    local bdf drv
    bdf=$(detect_pci_bdf)
    drv=$(current_driver "$bdf")

    if [[ "$drv" == "nvidia" ]]; then
        ok "already bound to nvidia; nothing to do"
        return 0
    fi

    confirm "About to unbind vfio-pci at $bdf and load nvidia stack. Sure?"

    if [[ "$drv" == "vfio-pci" ]]; then
        run_sh "echo '$bdf' > /sys/bus/pci/devices/$bdf/driver/unbind"
    fi

    # Clear driver_override so nvidia can claim it.
    if [[ -e "/sys/bus/pci/devices/$bdf/driver_override" ]]; then
        run_sh "echo '' > /sys/bus/pci/devices/$bdf/driver_override"
    fi

    # Load nvidia. modprobe handles dep order.
    run modprobe nvidia
    # Optional extras — load only if installed; do not error if missing.
    for extra in nvidia_uvm nvidia_modeset nvidia_drm; do
        if modinfo "$extra" >/dev/null 2>&1; then
            run modprobe "$extra" || warn "modprobe $extra failed (non-fatal)"
        fi
    done

    # nvidia driver auto-binds on modprobe via its pci_driver registration;
    # usually no explicit bind needed. But if not bound, try.
    if [[ "$(current_driver "$bdf" || true)" != "nvidia" ]]; then
        if [[ -d /sys/bus/pci/drivers/nvidia ]]; then
            run_sh "echo '$bdf' > /sys/bus/pci/drivers/nvidia/bind" || true
        fi
    fi

    # Verify via nvidia-smi (authoritative).
    if command -v nvidia-smi >/dev/null 2>&1; then
        if (( DRY_RUN )); then
            log "[dry] would run: nvidia-smi --list-gpus"
        else
            if nvidia-smi --list-gpus; then
                ok "nvidia-smi working"
            else
                die "nvidia-smi failed after rebind"
            fi
        fi
    else
        warn "nvidia-smi not in PATH; cannot verify"
    fi
}

# ---------------------------------------------------------------------------
# chmod — keep nvidia, just widen sysfs resource perms (racy with nvidia!)
# ---------------------------------------------------------------------------
#
# This is the "lightweight" path. It leaves nvidia.ko bound — /dev/nvidia*
# stays functional, vLLM stays up — but relaxes permissions on the PCI
# resource sysfs files so a non-root userspace probe can mmap BAR0/BAR4
# *read-only* for inspection. For actual GPU control (writes to BAR0) you
# still need vfio-pci or the nvidia uAPI; concurrent BAR0 writes from
# userspace and from nvidia.ko race and WILL eventually wedge the device.
# Use this only for read-only probes (e.g. cbuf0_probe, bar_map dry runs).

cmd_chmod() {
    local bdf
    bdf=$(detect_pci_bdf)

    local grp="$TARGET_GROUP"
    if [[ -z "$grp" ]]; then
        grp=$(id -gn)
    fi
    if ! getent group "$grp" >/dev/null 2>&1; then
        die "group '$grp' not found on this system"
    fi

    if ! id -nG "$USER" 2>/dev/null | tr ' ' '\n' | grep -qx "$grp"; then
        warn "current user '$USER' is not in group '$grp'; you will not gain access from this shell"
    fi

    for r in resource0 resource1 resource3 resource4; do
        local path="/sys/bus/pci/devices/$bdf/$r"
        [[ -e "$path" ]] || continue
        log "chgrp $grp $path && chmod 660 $path"
        run chgrp "$grp" "$path"
        run chmod 660 "$path"
    done

    warn "nvidia.ko is still bound — only use this for READ-ONLY probing."
    warn "Concurrent BAR writes from nvidia.ko and userspace will wedge the GPU."
}

# ---------------------------------------------------------------------------
# doctor — verbose dump
# ---------------------------------------------------------------------------

cmd_doctor() {
    local bdf
    bdf=$(detect_pci_bdf)
    echo "=== host ==="
    uname -a
    echo
    echo "=== PCI device $bdf ==="
    lspci -vv -s "${bdf#0000:}" 2>/dev/null || true
    echo
    echo "=== lspci -k ==="
    lspci -k -s "${bdf#0000:}" 2>/dev/null || true
    echo
    echo "=== sysfs ==="
    ls -l "/sys/bus/pci/devices/$bdf/" 2>/dev/null | head -n 40 || true
    echo
    echo "    driver link: $(readlink -f "/sys/bus/pci/devices/$bdf/driver" 2>/dev/null || echo '<none>')"
    echo "    driver_override: $(cat "/sys/bus/pci/devices/$bdf/driver_override" 2>/dev/null || echo '<unset>')"
    echo
    echo "=== lsmod (nvidia*/vfio*) ==="
    lsmod | awk 'NR==1 || $1 ~ /^(nvidia|vfio)/'
    echo
    echo "=== /dev/nvidia* ==="
    ls -l /dev/nvidia* 2>/dev/null || echo "<none>"
    echo
    echo "=== openers of /dev/nvidia* ==="
    if command -v fuser >/dev/null 2>&1; then
        fuser -v /dev/nvidia* 2>&1 || true
    fi
    echo
    echo "=== :8001 listeners ==="
    ss -tlnp 2>/dev/null | awk 'NR==1 || $4 ~ /:8001$/'
    echo
    echo "=== nvidia-smi ==="
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --list-gpus 2>&1 || true
    else
        echo "not installed"
    fi
    echo
    echo "=== relevant kernel tainting/logs (last 30) ==="
    dmesg 2>/dev/null | tail -n 200 \
        | grep -Ei 'nvidia|vfio|iommu' | tail -n 30 || true
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

require_root_for() {
    local sub="$1"
    case "$sub" in
        unbind|rebind|chmod)
            if [[ "$EUID" -ne 0 ]]; then
                die "$sub requires root (try: sudo $0 $sub)"
            fi
            ;;
    esac
}

main() {
    parse_args "$@"
    require_root_for "$SUBCMD"
    case "$SUBCMD" in
        status) cmd_status;;
        unbind) cmd_unbind;;
        rebind) cmd_rebind;;
        chmod)  cmd_chmod;;
        doctor) cmd_doctor;;
        *) die "unreachable";;
    esac
}

main "$@"

# ---------------------------------------------------------------------------
# KNOWN LIMITATIONS (documented for operator awareness):
#
# 1. This script does not handle the case where the GPU is in an IOMMU group
#    with other devices. For proper VFIO use the entire group must be bound
#    to vfio-pci. On GH200 the GPU is usually alone in its group, but verify
#    with `ls /sys/bus/pci/devices/$BDF/iommu_group/devices/` before relying
#    on DMA from userspace.
#
# 2. `rebind` does not restore nvidia-persistenced automatically. Run
#    `systemctl start nvidia-persistenced` manually if you want it back.
#
# 3. `chmod` widens sysfs resource* files only. It does not touch
#    /dev/mem, /proc/bus/pci, or VFIO char device permissions — those
#    paths are not used by this route.
#
# 4. TODO(verify): The exact set of nvidia_* modules that exist on this
#    kernel is discovered at runtime via lsmod; the script does not
#    assume nvidia_peermem / nvidia_uvm / nvidia_modeset / nvidia_drm
#    are all present. If your system has out-of-tree modules (e.g.
#    nvidia-fs, gdrdrv) that depend on nvidia.ko, add them to your
#    unload script BEFORE invoking this one — lsmod topo sort handles
#    dependencies but only among modules whose names start nvidia/nvidia_.
#
# 5. TODO(verify): `new_id` writes are idempotent on most kernels but can
#    error with EEXIST on some. We ignore that non-fatally via the
#    subsequent explicit /bind write. If your kernel behaves differently
#    (e.g. 6.8+ stricter handling), adjust the run_sh lines.
#
# 6. TODO(verify): On GH200 specifically, the docs call out that BAR4 is
#    "coherent HBM" (gsp-native.md §8). This script does not touch BAR4
#    addressing or NUMA policy. If userspace probes need specific mbind/
#    numactl setup, that's a separate tool.
# ---------------------------------------------------------------------------
