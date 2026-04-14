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
# NOT enabled: attempting to link them today fails with
#   fsp/mctp.o: undefined reference to `fsp_queue_advance_head'
# because fsp/mctp.s at (.text+0x1f4) calls `fsp_queue_advance_head`, but
# fsp/emem_xfer.s:177 only exports `fsp_queue_advance`. The ~3,225 LOC of
# FSP code in fsp/*.s was drafted but never linked, so it never got a clean
# symbol audit. Do not uncomment the block below until the symbol mismatch
# is resolved (rename one to match the other, or split the function) AND
# the unused fsp/*.s files are proven to link in isolation.
#
# Symbol inventory (per grep on globals, 2026-04-14):
#   fsp/bootcmd.s      : fsp_send_boot_commands
#   fsp/cot_payload.s  : cot_extract_hash, cot_extract_pubkey,
#                        cot_extract_signature, cot_build_payload
#   fsp/diag.s         : print_hex32, fsp_diag_dump_scratch,
#                        fsp_diag_dump_queues, fsp_diag_decode_error
#   fsp/emem_xfer.s    : fsp_emem_write, fsp_emem_read, fsp_queue_advance
#   fsp/fmc_load.s     : fmc_image_cpu/phys/size, fmc_hash_*, fmc_pubkey_*, fmc_sig_cpu
#   fsp/init.s         : fsp_init
#   fsp/mctp.s         : (no .global exports; references fsp_queue_advance_head)
#   fsp/registers.s    : (no .global exports; equ-only)
#   fsp/response.s     : fsp_response_poll, fsp_response_parse, fsp_response_ack
#
#SRCS += fsp/registers.s
#SRCS += fsp/init.s
#SRCS += fsp/emem_xfer.s
#SRCS += fsp/mctp.s
#SRCS += fsp/cot_payload.s
#SRCS += fsp/response.s
#SRCS += fsp/fmc_load.s
#SRCS += fsp/diag.s
#SRCS += fsp/bootcmd.s

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
