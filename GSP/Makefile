# GSP boot — ARM64 assembly, no libc, raw syscalls.
# Minimal Makefile: `as` to assemble, `ld` to link.

AS      := as
LD      := ld
ASFLAGS := --fatal-warnings

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
SRCS += gsp_common.s
SRCS += boot.s

# FSP (Step 7) sources.
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

# Test probe binary — exercises Steps 1 & 2 (bar_map + pmc_check) only.
TEST_PROBE_SRCS := test_probe.s bar_map.s pmc_check.s gsp_common.s
TEST_PROBE_OBJS := $(TEST_PROBE_SRCS:.s=.o)
TEST_PROBE_BIN  := gsp_probe

# Test alloc binary — exercises hbm_alloc arithmetic (no hardware needed).
TEST_ALLOC_SRCS := test_alloc.s hbm_alloc.s gsp_common.s
TEST_ALLOC_OBJS := $(TEST_ALLOC_SRCS:.s=.o)
TEST_ALLOC_BIN  := test_alloc

# Test falcon binary — read-only Falcon register probe.
TEST_FALCON_SRCS := test_falcon_read.s bar_map.s gsp_common.s
TEST_FALCON_OBJS := $(TEST_FALCON_SRCS:.s=.o)
TEST_FALCON_BIN  := test_falcon

# Emergency reset binary — standalone falcon recovery tool.
RESET_SRCS := reset_main.s bar_map.s gsp_common.s falcon_reset.s
RESET_OBJS := $(RESET_SRCS:.s=.o)
RESET_BIN  := gsp_reset

# Dry-run probe binaries — safe validation tools.
# probe_bcr_dry: programs BCR registers, reads back, skips CPUCTL (no RISC-V start)
# probe_fsp_dry: writes/reads EMEM, skips QUEUE_HEAD advance (FSP not notified)
PROBE_COMMON_OBJS := bar_map.o pmc_check.o falcon_reset.o hbm_alloc.o fw_load.o \
	bcr_start.o poll_lockdown.o rpc_channel.o gsp_common.o \
	fsp/registers.o fsp/init.o fsp/emem_xfer.o fsp/mctp.o \
	fsp/cot_payload.o fsp/response.o fsp/fmc_load.o fsp/diag.o fsp/bootcmd.o

.PHONY: all clean check test reset probes
all: $(BIN)

# Link all .o into the final executable. No libc.
$(BIN): $(OBJS)
	$(LD) -o $@ $(OBJS)

# Test binaries (probe and falcon require root + GPU; alloc is pure arithmetic).
test: $(TEST_PROBE_BIN) $(TEST_ALLOC_BIN) $(TEST_FALCON_BIN)

$(TEST_PROBE_BIN): $(TEST_PROBE_OBJS)
	$(LD) -o $@ $(TEST_PROBE_OBJS)

$(TEST_ALLOC_BIN): $(TEST_ALLOC_OBJS)
	$(LD) -o $@ $(TEST_ALLOC_OBJS)

$(TEST_FALCON_BIN): $(TEST_FALCON_OBJS)
	$(LD) -o $@ $(TEST_FALCON_OBJS)

# Emergency reset binary (standalone falcon recovery).
reset: $(RESET_BIN)

$(RESET_BIN): $(RESET_OBJS)
	$(LD) -o $@ $(RESET_OBJS)

# Dry-run probe binaries.
probes: probe_bcr_dry probe_fsp_dry

probe_bcr_dry: probe_bcr_dry.o $(PROBE_COMMON_OBJS)
	$(LD) -o $@ probe_bcr_dry.o $(PROBE_COMMON_OBJS)

probe_fsp_dry: probe_fsp_dry.o $(PROBE_COMMON_OBJS)
	$(LD) -o $@ probe_fsp_dry.o $(PROBE_COMMON_OBJS)

# Assemble each .s into .o. Rebuilds if the .s changes.
%.o: %.s gsp_common.s
	$(AS) $(ASFLAGS) -o $@ $<

# Syntax check: assemble every source to /dev/null.
ALL_SRCS := $(sort $(SRCS) $(TEST_PROBE_SRCS) $(TEST_ALLOC_SRCS) $(TEST_FALCON_SRCS) $(RESET_SRCS) probe_bcr_dry.s probe_fsp_dry.s)
check:
	@set -e; for f in $(ALL_SRCS); do \
		echo "check $$f"; \
		$(AS) $(ASFLAGS) -o /dev/null $$f; \
	done
	@echo "All $(words $(ALL_SRCS)) sources pass syntax check."

clean:
	rm -f $(OBJS) $(TEST_PROBE_OBJS) $(TEST_ALLOC_OBJS) $(TEST_FALCON_OBJS) $(RESET_OBJS) \
		$(BIN) $(TEST_PROBE_BIN) $(TEST_ALLOC_BIN) $(TEST_FALCON_BIN) $(RESET_BIN) \
		probe_bcr_dry.o probe_bcr_dry probe_fsp_dry.o probe_fsp_dry
