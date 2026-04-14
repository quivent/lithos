# GSP boot — ARM64 assembly, no libc, raw syscalls.
# Minimal Makefile: `as` to assemble, `ld` to link.

AS  := as
LD  := ld

# Step files (library-style routines).
SRCS := \
	bar_map.s \
	pmc_check.s \
	falcon_reset.s \
	hbm_alloc.s \
	fw_load.s \
	bcr_start.s \
	poll_lockdown.s \
	rpc_channel.s

# Common string/message pool + the _start driver that calls the 9 steps.
# Previously commented out with "Enable these once they exist" — they exist.
SRCS += gsp_common.s
SRCS += boot.s

# FSP (Step 7) sources — Firmware Security Processor boot-command exchange.
# ~3,225 LOC. Links clean as of 2026-04-14 after renaming
# fsp_queue_advance → fsp_queue_advance_head in fsp/emem_xfer.s (matches
# the .extern declaration in fsp/mctp.s:73 and the .equ NV_PFSP_QUEUE_HEAD
# register it writes to).
#
# NOTE: linking these files makes fsp_send_boot_commands callable, but
# boot.s:123 still prints the TODO placeholder at step 7 — the dispatch
# (replace the TODO with `bl fsp_send_boot_commands`) is a separate
# commit, gated on a dry-run validation plan (see docs/runtime/
# gsp-hard-test-protocol.md).
#
# Symbol inventory (per grep on globals):
#   fsp/bootcmd.s      : fsp_send_boot_commands
#   fsp/cot_payload.s  : cot_extract_hash, cot_extract_pubkey,
#                        cot_extract_signature, cot_build_payload
#   fsp/diag.s         : print_hex32, fsp_diag_dump_scratch,
#                        fsp_diag_dump_queues, fsp_diag_decode_error
#   fsp/emem_xfer.s    : fsp_emem_write, fsp_emem_read, fsp_queue_advance_head
#   fsp/fmc_load.s     : fmc_image_cpu/phys/size, fmc_hash_*, fmc_pubkey_*, fmc_sig_cpu
#   fsp/init.s         : fsp_init
#   fsp/mctp.s         : mctp_packetize, mctp_depacketize
#                        (calls fsp_queue_advance_head via .extern)
#   fsp/registers.s    : equ-only register map
#   fsp/response.s     : fsp_response_poll, fsp_response_parse, fsp_response_ack
#
SRCS += fsp/registers.s
SRCS += fsp/init.s
SRCS += fsp/emem_xfer.s
SRCS += fsp/mctp.s
SRCS += fsp/cot_payload.s
SRCS += fsp/response.s
SRCS += fsp/fmc_load.s
SRCS += fsp/diag.s
SRCS += fsp/bootcmd.s

OBJS := $(SRCS:.s=.o)

BIN := gsp_boot

.PHONY: all clean check
all: $(BIN)

# Link all .o into the final executable. No libc.
$(BIN): $(OBJS)
	$(LD) -o $@ $(OBJS)

# Assemble each .s into .o. Rebuilds if the .s changes.
%.o: %.s
	$(AS) -o $@ $<

# Syntax check: assemble each source to /dev/null.
check:
	@set -e; for f in $(SRCS); do \
		echo "check $$f"; \
		$(AS) -o /dev/null $$f; \
	done

clean:
	rm -f $(OBJS) $(BIN)
