# GSP Hard-Test Protocol

The hard test attempts an end-to-end 9-step GSP boot: `bar_map_init` →
`pmc_check` → `falcon_reset` → `hbm_alloc_init` → `gsp_fw_load` →
`gsp_bcr_start` → **`fsp_send_boot_commands`** → `gsp_poll_lockdown` →
`gsp_rpc_alloc_channel`. Unlike the soft test (`tools/gsp-soft-test.sh`),
the hard test writes to GPU registers and can wedge the device.

**The hard test is not safe to run today.** This document captures what
must be true before attempting it, and is the canonical checklist for
whoever gets there.

---

## 1. What "hard test" means

- Runs `GSP/gsp_boot` as `_start` (entry `0x401040` after commit `58346de`).
- Writes to BAR0 registers including Falcon reset, BCR program, FSP
  EMEM/queue, PRIV_LOCKDOWN clear-polling, and RPC channel creation.
- On success: produces a live GPU channel with a valid RPC handle.
- On failure: can leave the Falcon in a stuck state requiring reboot
  (or, for some faults, power-cycle).

## 2. Prerequisites (must be true; no exceptions)

### 2.1 FSP call site is wired

- [ ] `GSP/boot.s:123` TODO replaced with `bl fsp_send_boot_commands`
      (and error-code handling on its `w0` return).
- [ ] Return code checked: non-zero from FSP must abort with a printed
      reason, not fall through to `gsp_poll_lockdown` (which will time
      out in 30 s without FSP).
- [ ] Verification: build cleanly, no undefined references.
      `cd GSP && make` must print no `ld: warning` lines.

### 2.2 FSP firmware is available

- [ ] `/lib/firmware/nvidia/<driver-version>/gsp_ga10x.bin` (or similar)
      present on disk. Check what `fw_load.s:1` hardcodes.
- [ ] File readable by the user running `gsp_boot` (probably root).

### 2.3 Host is prepped

- [ ] `tools/gpu-setup.sh status` shows nvidia bound, vLLM up, sysfs
      resource files root-only. This is the pre-state we revert to.
- [ ] `tools/gpu-setup.sh doctor` clean.
- [ ] `dmesg -T` in the last 5 minutes has no NVRM errors from prior runs.
- [ ] Free disk space for captured output: at least 500 MB.

### 2.4 Validation: soft test has run and passed recently

- [ ] `tools/gsp-soft-test.sh` returned 0 within the last hour on the
      same host. Logs in `logs/gsp-soft-test/`.
- [ ] `probe_bar0` output confirmed: PMC_BOOT_0 bits [23:20] = `0xa`
      (Hopper architecture marker).
- [ ] `probe_gsp_falcon` output: FALCON_HWCFG2 bit 13 observed — note
      whether lockdown is already clear (SBIOS pre-seed) or asserted.
      If asserted, FSP is MANDATORY; if clear, FSP is optional (but
      recommended for correctness).

### 2.5 Recovery path rehearsed

- [ ] `tools/gpu-setup.sh rebind` tested and works (rehearsed during
      the soft test).
- [ ] You have another shell open and can run `sudo ./tools/gpu-setup.sh
      rebind` without navigating away.
- [ ] You have IPMI / serial console access, or know who to call.
      Rebooting a stuck GH200 may require out-of-band power control.

### 2.6 vLLM state is understood

- [ ] Coordinate downtime. vLLM will be offline for (a) the soft test
      duration, (b) the hard test duration, (c) ~3–5 min CUDA graph
      JIT warmup on rebind + restart. Budget 20+ minutes of downtime.
- [ ] Hive workers that depend on `:8001` are paused or told to expect
      timeouts.

---

## 3. The protocol

### 3.1 Before you start

```bash
# confirm you're on the right machine
hostname
lspci -D | grep -i nvidia
nvidia-smi --query-gpu=name --format=csv,noheader

# confirm soft test passed recently
ls -lt logs/gsp-soft-test/*.log | head -1
tail -20 "$(ls -t logs/gsp-soft-test/*.log | head -1)"

# confirm BDF in bar_map.s matches this host (should match `lspci -D`)
grep -n "ascii \"" GSP/bar_map.s | head -3
```

### 3.2 Kill vLLM and unbind nvidia

```bash
pkill -f "vllm.entrypoints"
sleep 3
ss -tlnp | grep 8001 || echo "vllm gone"
sudo tools/gpu-setup.sh unbind --yes
sudo tools/gpu-setup.sh status   # confirm: nvidia unbound
```

### 3.3 Pre-flight register read (sanity)

```bash
# expect PMC_BOOT_0[23:20] = 0xa
sudo GSP/probe_bar0 2>&1 | tee logs/gsp-hard-test/pre-probe_bar0.log

# check initial PRIV_LOCKDOWN state (bit 13 of HWCFG2)
sudo GSP/probe_gsp_falcon 2>&1 | tee logs/gsp-hard-test/pre-probe_falcon.log
```

If either of these fails or the values don't match expectations,
**abort**: run `sudo tools/gpu-setup.sh rebind` and investigate.

### 3.4 Run the hard test

```bash
# capture dmesg baseline
sudo dmesg --time-format iso > logs/gsp-hard-test/pre-dmesg.log

# THIS IS THE DESTRUCTIVE CALL. Watch the output.
sudo GSP/gsp_boot 2>&1 | tee logs/gsp-hard-test/gsp_boot.log
EXIT_CODE=$?

# capture post-state regardless
sudo dmesg --time-format iso > logs/gsp-hard-test/post-dmesg.log
```

Expected output is the 9-step banner sequence from `GSP/boot.s`. Each
step prints `gsp: [N/9] <name> -- begin` and `gsp: [N/9] <name> -- OK`
(or `FAILED`). Exit codes (from `boot.s:38` header):

- `0` — success, all 9 steps passed
- `1..9` — which step failed (failure codes documented in `boot.s`)

### 3.5 Recover

Whether the hard test succeeded or failed:

```bash
sudo tools/gpu-setup.sh rebind
sudo tools/gpu-setup.sh status    # confirm nvidia bound
nvidia-smi                        # confirm device healthy

# restart vLLM if the box is shared
nohup /home/ubuntu/launch_vllm.sh > /home/ubuntu/vllm.log 2>&1 &
# wait ~3-5 min for JIT, then
curl -sf http://localhost:8001/v1/models | python3 -m json.tool
```

If `nvidia-smi` hangs or returns an error after rebind: you're in the
failure mode that requires **reboot**. Do not repeat the rebind more
than once; escalate.

### 3.6 Success signal

The hard test has "worked" when:

- `gsp_boot` exit code = `0`
- `nvidia-smi` healthy after rebind
- `diff pre-dmesg.log post-dmesg.log` shows no kernel errors (some
  NVRM re-initialization messages on rebind are normal)
- `gsp_boot.log` shows all 9 steps with `OK`

Partial success (some steps OK, step N failed) is useful data — commit
the logs to a record file; do not try to "continue from where it left
off." Always run the full sequence from `_start`.

---

## 4. Known failure modes (from the risk matrix)

Summary (full list in `docs/runtime/gsp-bringup-runbook.md` §5):

| Signature | Likely cause | Recovery |
|---|---|---|
| `gsp_boot` hangs at step 3 | Falcon reset stuck; bad HWCFG2 assumptions | Reboot — no soft recovery |
| Step 5 fails: `firmware mmap failed` | Missing `/lib/firmware/nvidia/.../gsp_ga10x.bin` | Fix firmware path, re-run |
| Step 7 times out after ~5 s | FSP queue not responsive; EMEM handshake failed | Rebind + reboot GPU only if nvidia-smi hangs |
| Step 8 times out after 30 s | PRIV_LOCKDOWN never cleared — FSP didn't authorize | Means step 7 silently failed; review FSP diagnostic output |
| SIGBUS on our process | Wrote to an unmapped/RO BAR0 offset | GPU state probably clean; rebind should work |
| `nvidia-smi` hangs after rebind | GPU wedged by destructive write | Reboot |
| SSH becomes slow/unresponsive | GH200 fabric wedged (rare) | Power cycle via IPMI |

---

## 5. After a successful hard test

- [ ] Commit `logs/gsp-hard-test/<timestamp>/` to the repo as the
      reference "known-good" trace — the first such artifact the repo
      has ever had.
- [ ] Update `STATUS.md`:  
      `HARDWARE [████████]` once end-to-end is proven.
- [ ] Update this document's §2.4 expected register values with the
      actual observed values (PMC_BOOT_0, HWCFG2, scratch groups).

---

## 6. After a failed hard test

- [ ] Preserve `logs/gsp-hard-test/<timestamp>/` — do not delete.
- [ ] `grep 'FAILED\|ERROR' gsp_boot.log` — identify which step.
- [ ] Compare against the per-step expected output in `boot.s` comments.
- [ ] Update §2.4 of this doc with the failure signature so the next
      attempt starts informed.
- [ ] Do not attempt again until a code change addresses the root
      cause. "Try again and see" after a GPU-state-changing failure
      is how devices get wedged permanently.

---

## 7. References

- `GSP/boot.s` — the 9-step driver
- `GSP/gsp_common.s` — register table, timeouts, message pool
- `GSP/fsp/bootcmd.s` — `fsp_send_boot_commands` entry
- `tools/gpu-setup.sh` — unbind/rebind utility
- `tools/gsp-soft-test.sh` — the read-only pre-flight
- `docs/runtime/gsp-bringup-runbook.md` — per-step detail
- `docs/runtime/gsp-native.md` — original design
- `GSP/fsp_plan.md` — FSP implementation plan (historical; the code
  exists now, see `GSP/fsp/*.s`)
