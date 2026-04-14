# GSP Bringup Runbook

## 0. What this is for

This runbook is for the operator attempting to drive the NVIDIA GSP (GPU
System Processor) on this GH200 from userspace using the Lithos assembly
sequence in `GSP/`. Read it before any attempt that touches the GPU
while the `nvidia` driver is bound or while vLLM is serving on
`:8001`. The expected outcome of a full run is that the nine boot
stages in `GSP/boot.s:1-41` complete in order, with step 6 starting
the RISC-V FMC, step 8 observing `PRIV_LOCKDOWN` clear, and step 9
round-tripping a first RPC. Anything short of that is a partial
bringup; this runbook also tells you how to stop cleanly at each
partial state without wedging the machine.

## 1. Pre-flight checklist

Work top to bottom. Do not skip. Every item has a verification
command and a pass criterion tied to either source or host state.

- [ ] GPU BDF matches what the assembly hard-codes. Read-only probe
  files target `0000:dd:00.0` (`GSP/probe_pci.s:25`,
  `GSP/probe_bar0.s:41-42`, `GSP/probe_fsp.s:39`). The takeover path
  in `GSP/bar_map.s:67-69` hard-codes `0000:09:00.0` and warns that
  the BDF slot is duplicated in three places
  (`GSP/bar_map.s:15-21`). Verify with `lspci -d 10de:` and patch
  both sets of strings before building if your BDF differs.
- [ ] `lspci -k -s dd:00.0` shows `Kernel driver in use: nvidia` (or
  `vfio-pci` if already unbound on a prior run).
- [ ] `nvidia-smi` returns within 5 s without errors. If it hangs, do
  not proceed — an unresponsive driver will poison any BAR0 probe.
- [ ] No CUDA processes holding nvidia: `lsof /dev/nvidia*` returns
  nothing except expected service daemons.
- [ ] vLLM state documented. If `ss -ltn | grep :8001` shows a
  listener, path A (read-only) is still safe; path B requires taking
  it down first (see section 2).
- [ ] `dmesg -T --since "5 minutes ago"` is clean of NVRM / Xid /
  FALCON errors. Any pre-existing GSP complaint will confuse
  post-run triage.
- [ ] Lithos bootstrap binary exists: `ls bootstrap/lithos-bootstrap`.
  This is the self-hosted Lithos tool referenced by `STATUS.md` at
  line 44. It is not strictly required for running the GSP probes —
  those are independent `.o` files built from `GSP/Makefile:22-35` —
  but a broken bootstrap usually means the environment is off.
- [ ] Firmware blob is present at the path `GSP/fw_load.s:46` expects:
  `/lib/firmware/nvidia/580.105.08/gsp_ga10x.bin`. `gsp_fw_load`
  calls `SYS_EXIT` on failure with no return
  (`GSP/boot.s:260-261`), so a missing blob aborts step 5 hard.
- [ ] GSP prebuilt artefacts are present or rebuildable: `ls
  GSP/gsp_probe GSP/gsp_boot`. If absent, `cd GSP && make` regenerates
  from `GSP/Makefile:26-35`.

## 2. Decision tree

### Path A: Keep `nvidia` bound, read-only validation only

Choose path A whenever vLLM on `:8001` is in use, when you have not
yet confirmed that the takeover sequence will recover cleanly, or
when you only need to confirm the GPU is alive and FSP has booted.

Path A uses only the probe files, which open PCI sysfs
`resource*` files read-only or read/write without touching any
write-sensitive register. Safe probes:

- `GSP/probe_pci.s` — PCI config-space header, BARs, link state. All
  reads through sysfs `config` / `resource` / `numa_node` /
  `current_link_speed` / `current_link_width`
  (`GSP/probe_pci.s:24-33`). No MMIO.
- `GSP/probe_bar0.s` — mmaps BAR0 and reads `PMC_BOOT_0`,
  `PMC_BOOT_42`, `PMC_ENABLE`, `PTIMER_TIME_{0,1}`,
  `THERM_I2CS_SCRATCH`, `FALCON_ENGINE`, `FALCON_HWCFG2`
  (`GSP/probe_bar0.s:26-33`). These are all reads; no stores.
- `GSP/probe_bar4.s` — maps the BAR4 coherent window for sanity
  reads.
- `GSP/probe_fsp.s` — reads FSP scratch groups 2/3 and all four
  queue head/tail pairs (`GSP/probe_fsp.s:27-33`). Read-only.
- `GSP/probe_gsp_falcon.s`, `GSP/probe_ce.s`, `GSP/probe_sm.s`,
  `GSP/probe_ptimer.s`, `GSP/probe_mem.s` — all read-only per their
  headers.

**What path A cannot confirm:** that `falcon_reset.s` or
`bcr_start.s` will work. Those require the driver unbound
(`GSP/bar_map.s:27-28` documents the requirement).

### Path B: Full unbind + takeover

Choose path B only when you are ready to lose vLLM for the duration
of the attempt plus a cold-start window on rebind (JIT recompilation
of CUDA caches typically takes minutes). Path B runs the full nine-
step sequence in `GSP/boot.s:163-394`.

Cost summary:

- vLLM on `:8001` must be stopped before unbind. It will need to be
  restarted after rebind.
- Step 7 (FSP Chain-of-Trust) is documented as `TODO -- NOT
  IMPLEMENTED` at `GSP/boot.s:10, 122-124` and `GSP/README.md:57-85`.
  The `GSP/fsp/` subdirectory contains work in progress (`init.s`,
  `bootcmd.s`, `response.s`, etc.) but is not yet wired into
  `boot.s`. **VERIFY**: whether `GSP/fsp/*.s` are linked into the
  current `gsp_boot` — grep `GSP/Makefile` for `fsp/`. Sibling
  agent is resolving the FSP UNSTARTED-vs-DONE conflict; defer to
  their write-up rather than assuming one way.
- Without step 7, `poll_lockdown.s` will time out after 30 s
  (`GSP/gsp_common.s:140`, `GSP/poll_lockdown.s:43`), yielding exit
  code 4 per `GSP/boot.s:38`.

## 3. Path A runbook

### Step A1 — Build the probes if not already built

Goal: produce standalone probe ELFs. Command: `cd GSP && make` (rule
at `GSP/Makefile:30-35`). Expected: `gsp_probe` and the individual
`probe_*` binaries present. Failure signature: `as` errors on any
`.s`; re-read the offending file and check syntax.

### Step A2 — PCI identity and link

Goal: confirm the device is visible and the link is up. Command:
`sudo ./GSP/probe_pci`. Expected: the "NVIDIA (0x10DE)" branch in
`GSP/probe_pci.s:78-79` fires, BAR sizes match the Hopper layout.
Failure: vendor mismatch means wrong BDF — see pre-flight item 1.

### Step A3 — PMC identity

Goal: confirm BAR0 MMIO works and this is Hopper. Command: `sudo
./GSP/probe_bar0`. Expected output includes `Architecture = HOPPER
(0xA)`, per `GSP/probe_bar0.s:71-72` and
`GSP/gsp_common.s:86-87`. `PMC_BOOT_0` bits[23:20] == 0xA is the
check `pmc_check` itself performs (`GSP/pmc_check.s:14-28`).
Failure signatures:

- `PMC_BOOT_0 == 0xffffffff`: BAR0 reads returning all-ones means
  the mmap landed on an unbacked region or the device is in reset.
- `Architecture != 0xA`: wrong GPU; abort before any path B step.

### Step A4 — FSP sanity

Goal: confirm FSP has booted and no latched error. Command: `sudo
./GSP/probe_fsp`. Expected: low byte of `THERM_I2CS_SCRATCH` is
`0xFF` (`GSP/probe_bar0.s:77-80`, "FSP boot = COMPLETE"), and
`SCRATCH_GROUP_3_0` prints all-zero (the condition `fsp_init`
enforces at `GSP/fsp/init.s:27-35`). Failure: non-zero
`SCRATCH_GROUP_3` means the FSP has latched a prior fault — do
**not** run path B until that clears (typically requires an SBIOS
reset).

### Step A5 — Read-only falcon and queue state

Goal: snapshot GSP Falcon state before any attempt. Command: `sudo
./GSP/probe_gsp_falcon`. Inspect `FALCON_ENGINE` (`0x1103C0`) and
`FALCON_HWCFG2` (`0x1100F4`) per `GSP/gsp_common.s:90-101`. The
`PRIV_LOCKDOWN` bit is bit 13 of HWCFG2 (`GSP/poll_lockdown.s:36`,
`GSP/gsp_common.s:101`). Record the current values; they are what
`poll_lockdown.s` will be racing against if you later run path B.

## 4. Path B runbook

### Step B0 — Prerequisites

Path B assumes path A completed without surprises and the operator
has decided the cost is acceptable. **VERIFY**: the current state of
the `tools/gpu-setup.sh` helper — a sibling agent is drafting it;
there is no such file at `tools/` today (confirmed: `ls tools/`
shows only the `cbuf0_probe`, `dump_qmd`, `qmd_probe_driver`
binaries and their sources). Do not invent an unbind sequence;
consume whatever that agent produces.

### Step B1 — Take vLLM down

Goal: free `:8001` and any processes holding `/dev/nvidia*`.
Verification: `ss -ltn | grep :8001` empty, `lsof /dev/nvidia*`
empty. Failure to fully drain will make the unbind fail or, worse,
appear to succeed while leaving stale mappings.

### Step B2 — Unbind `nvidia` from the target device

Deferred to the `tools/gpu-setup.sh` script currently being drafted.
Do not improvise this with raw `/sys/bus/pci/drivers/nvidia/unbind`
echoes without reviewing that script, because the order of
`nvidia-uvm` / `nvidia-drm` / `nvidia` module teardown matters and
getting it wrong can leave the kernel with dangling references.

### Step B3 — Invoke the 9-step boot

Goal: run `gsp_boot` end to end. Command: `sudo ./GSP/gsp_boot`.
The binary is the linked output of `GSP/Makefile:30-31`; entry point
is `_start` at `GSP/boot.s:162-163`.

Expected stage-by-stage banners come from the `msg_step*_begin`
strings at `GSP/boot.s:88-149`. Each begins `gsp: [N/9]` and
terminates with either `-- OK` or `-- FAILED, aborting`.

Per-step failure signatures and exit codes (`GSP/boot.s:33-40`):

- `[1/9] bar_map_init -- FAILED`: exit 1. BAR sysfs open or mmap
  failed. Either nvidia is still bound (check step B2) or the BDF
  in `GSP/bar_map.s:67-69` does not match your hardware.
- `[2/9] pmc_check -- FAILED`: exit 2. Not Hopper, or BAR0 not
  mapped (`GSP/pmc_check.s:36-43`).
- `[3/9] falcon_reset -- FAILED`: exit 3. Reset poll timed out
  after ~100M iterations (`GSP/falcon_reset.s:46-47`,
  `GSP/gsp_common.s:141`). Recovery requires rebind + full reset.
- `[4/9] hbm_alloc_init -- FAILED`: exit 6. Arguments bad; should
  not happen if step 1 succeeded.
- `[5/9] gsp_fw_load`: no OK path to fail cleanly — the routine
  `SYS_EXIT`s directly on any ELF parse or firmware-open error
  (`GSP/boot.s:260-261`). Check `/lib/firmware/nvidia/580.105.08/`.
- `[6/9] gsp_bcr_start`: no return check (`GSP/boot.s:312-317`).
  Fire and forget; success is inferred from step 8.
- (gap) Step 7 prints the TODO notice from
  `GSP/boot.s:122-124` if the placeholder is wired in; at present
  boot.s jumps directly from step 6 to step 8
  (`GSP/boot.s:317-324`), skipping any FSP handshake.
- `[8/9] gsp_poll_lockdown -- FAILED`: exit 4. Either 30 s timeout
  (`GSP/poll_lockdown.s:135-136`) or FMC error (`-2` return,
  `GSP/poll_lockdown.s:92-94`). See section 5.
- `[9/9] gsp_rpc_alloc_channel -- FAILED`: exit 5. Queue never
  serviced; likely cascades from step 8 even though step 8
  "succeeded" in some corner cases.

### Step B4 — Rebind

Goal: restore `nvidia` to the device. Use the same
`tools/gpu-setup.sh` script (reverse path). Expect a cold-start
window of several minutes before `nvidia-smi` returns cleanly and
before vLLM JIT caches warm.

## 5. Failure → recovery matrix

| Symptom | Soft recovery | Hard recovery | Reboot required |
|---|---|---|---|
| Step 1 fails, nvidia still bound | Unbind per B2, retry | — | No |
| Step 2 reports non-Hopper arch | Check BDF | — | No (misconfig) |
| Step 3 falcon reset timeout (`GSP/falcon_reset.s:46-47`) | Rebind, `nvidia-smi` idle, retry | Power-cycle GPU via SBIOS if exposed | Yes if Falcon stays stuck |
| Step 5 `SYS_EXIT` inside `gsp_fw_load` | Fix firmware path (`GSP/fw_load.s:46`) | — | No |
| Step 8 `-1` timeout | Rebind, inspect `dmesg` for FSP messages | Path B again once FSP step is implemented | Yes if HWCFG2 bit 13 never clears even after rebind |
| Step 8 `-2` FMC error (MAILBOX0 nonzero and != fmc_params_pa[31:0], `GSP/poll_lockdown.s:82-94`) | Decode MAILBOX0 against `GSP/fsp/response.s:38-44` status codes | Rebind nvidia | Yes if latched in SCRATCH_GROUP_3 (`GSP/fsp/init.s:31-35`) |
| Step 9 RPC never responds | Inspect queue head/tail via `probe_fsp` (even though it reads FSP queues, same pattern applies to `GSP_QUEUE_HEAD_BASE` 0x110C00, `GSP/gsp_common.s:118-119`) | Rebind | Only if lockdown bit is still stuck |
| Host hang during any step | sysrq if available; otherwise power cycle | — | Yes |

## 6. Post-run verification

Confirm Lithos actually moved the GPU. Signals, strongest first:

1. `PMC_BOOT_0` read back from BAR0+0x0 equals the expected Hopper
   identity with bits[23:20] == 0xA (`GSP/gsp_common.s:86-87`,
   `GSP/pmc_check.s:15-20`). If the prior probe saw this and the
   post-run probe still sees it, BAR0 was not corrupted.
2. `FALCON_HWCFG2` bit 13 clear after step 8 (`GSP/poll_lockdown.s:
   96-104`). This is the canonical "GSP is alive" signal.
3. `FALCON_ENGINE` at BAR0+0x1103C0 bits[10:8] == 0b010
   (DEASSERTED, `GSP/gsp_common.s:94-95`) after step 3.
4. Step 9 "OK" message from `GSP/boot.s:143-144` reaching stderr.
   That requires a real round-trip, not a fire-and-forget store.

The closest thing to a "hello world" run is `GSP/gsp_boot` itself —
the nine-step sequence is intentionally the smallest testable path.
There is no separate `lithos-stage1` GPU "hello world" in the tree
today; **VERIFY** by `ls bootstrap/ | grep stage1` before citing one.

## 7. Known gotchas (from prior runs)

- The `nvidia` driver auto-rebinds on reboot. Any unbind done by the
  sibling `tools/gpu-setup.sh` has to be re-run after every boot.
- GH200's coherent HBM through BAR4 is large; `bar_map.s` only maps
  a 256 MB initial window (`GSP/bar_map.s:48-51`,
  `GSP/gsp_common.s:131`), not the full window. Extending requires
  changing both `BAR4_SIZE_LO` and `bar_map.s`. Do not assume 128 GB
  of BAR4 VA is reachable from userspace without that change.
- `mmap` flag mismatches can fail silently if the kernel returns a
  partial mapping; the assembly does no post-mmap length check.
- `PRIV_LOCKDOWN` polling timeout is 30 s (`GSP/gsp_common.s:140`,
  `GSP/poll_lockdown.s:43, 115-116`). If FSP is not actually
  servicing the FMC, you will wait the full 30 s before seeing
  `FAILED`.
- `MAILBOX0` non-zero with a value that does not equal
  `fmc_params_pa[31:0]` is an FMC error code, not bootargs PA
  (`GSP/poll_lockdown.s:84-94`). Decode against the FSP status
  table at `GSP/fsp/response.s:38-44` before re-running.
- The BDF "09:00" ASCII fragment is duplicated three places in
  `GSP/bar_map.s` (noted at `GSP/bar_map.s:15-21` and
  `GSP/gsp_common.s:32-35`). Patching one copy and not the others
  silently references different devices.
- Step 7 (FSP Chain-of-Trust) is either unwired or partial — see the
  **VERIFY** in section 2 path B. Do not assume a clean step 8 on
  production silicon without it.

## 8. Reference — what each step talks to

All offsets are from `GSP/gsp_common.s` unless otherwise cited.

| Register | BAR0 offset | File:line | Purpose |
|---|---|---|---|
| `PMC_BOOT_0` | 0x000000 | `GSP/gsp_common.s:86` | Hopper ID (bits[23:20]==0xA); step 2 |
| `THERM_I2CS_SCRATCH` | 0x0200BC | `GSP/probe_bar0.s:31`, `GSP/probe_fsp.s:27` | FSP boot-complete sentinel (low byte 0xFF) |
| `FALCON_ENGINE` | 0x1103C0 | `GSP/gsp_common.s:90-95` | Falcon reset + status; step 3 |
| `FALCON_MAILBOX0` | 0x110040 | `GSP/gsp_common.s:98` | fmc_params_pa[31:0]; also FMC error code |
| `FALCON_MAILBOX1` | 0x110044 | `GSP/gsp_common.s:99` | fmc_params_pa[63:32] |
| `FALCON_HWCFG2` | 0x1100F4 | `GSP/gsp_common.s:100-101` | `PRIV_LOCKDOWN` at bit 13; step 8 |
| `BCR_DMACFG` | 0x11166C | `GSP/gsp_common.s:105` | BCR lock (0x80000001); step 6 |
| `BCR_PKCPARAM_LO/HI` | 0x111670 / 0x111674 | `GSP/gsp_common.s:106-107` | Manifest address; step 6 |
| `BCR_FMCCODE_LO/HI` | 0x111678 / 0x11167C | `GSP/gsp_common.s:108-109`, `GSP/bcr_start.s:62-67` | FMC code addr (>>3); step 6 |
| `BCR_FMCDATA_LO/HI` | 0x111680 / 0x111684 | `GSP/gsp_common.s:110-111`, `GSP/bcr_start.s:75-80` | FMC data addr (>>3); step 6 |
| `CPUCTL` | 0x111388 | `GSP/gsp_common.s:115` | STARTCPU bit 0; step 6 |
| `GSP_QUEUE_HEAD(0)` | 0x110C00 | `GSP/gsp_common.s:118` | RPC cmd head; step 9 |
| `GSP_QUEUE_TAIL(0)` | 0x110C04 | `GSP/gsp_common.s:119` | RPC cmd tail; step 9 |
| `USERD_BAR0_BASE` | 0xFC0000 | `GSP/gsp_common.s:122-124` | Channel doorbell base |
| `FSP_SCRATCH_GROUP_2(0)` | 0x8F0320 | `GSP/fsp/init.s:56`, `GSP/probe_fsp.s:28` | FSP boot-state sentinel |
| `FSP_SCRATCH_GROUP_3(0)` | 0x8F0330 | `GSP/fsp/init.s:57`, `GSP/probe_fsp.s:29` | FSP latched-error scratch |
| `FSP_QUEUE_HEAD/TAIL(0)` | 0x8F2C00 / 0x8F2C04 | `GSP/fsp/init.s:58-59`, `GSP/probe_fsp.s:30-31` | CPU→FSP cmd queue |
| `FSP_MSGQ_HEAD/TAIL(0)` | 0x8F2C80 / 0x8F2C84 | `GSP/fsp/init.s:60`, `GSP/response.s:29-30` | FSP→CPU msg queue |

Timeouts and constants:

| Name | Value | File:line |
|---|---|---|
| `TIMEOUT_SECS` (lockdown poll) | 30 | `GSP/gsp_common.s:140`, `GSP/poll_lockdown.s:43` |
| `POLL_TIMEOUT` (falcon reset) | 100000000 | `GSP/gsp_common.s:141`, `GSP/falcon_reset.s:47` |
| `RPC_POLL_LIMIT` | 10000000 | `GSP/gsp_common.s:142` |
| `BAR0_SIZE` | 16 MB | `GSP/gsp_common.s:130` |
| `BAR4_SIZE_LO` | 256 MB initial window | `GSP/gsp_common.s:131`, `GSP/bar_map.s:51` |
| `GSP_RESERVED_BYTES` | 64 MB at BAR4 base | `GSP/boot.s:49` |
| `ALIGN_2MB` | 0x200000 | `GSP/gsp_common.s:132`, `GSP/hbm_alloc.s:12` |

**VERIFY**: FSP step is authoritatively UNSTARTED vs partially
DONE — defer to the sibling agent's resolution before following this
runbook through to step 8 on a production boot.
