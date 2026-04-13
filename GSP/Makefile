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

# Enable these once they exist:
#SRCS += gsp_common.s
#SRCS += boot.s

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
