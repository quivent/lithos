\\ teardown.ls — replaces cuCtxDestroy + cuMemFree.
\\
\\ For an inference engine that runs once and exits, teardown is mostly
\\ symbolic — the kernel reclaims mappings and fds on process exit. But
\\ a long-running server (or a test harness that spins contexts up and
\\ down) needs to munmap the BAR windows and close the sysfs/vfio fd so
\\ the device can be rebound or reopened cleanly.
\\
\\ Mirrors runtime/init.ls:
\\   init opens resource0 / resource4 and mmaps BAR0 (16 MB) + BAR4 (128 GB)
\\   teardown munmaps both windows and closes the fd(s)
\\
\\ GSP graceful shutdown via RPC is left as a TODO — on process exit the
\\ GPU reaches a clean state through falcon reset on next bring-up.
\\
\\ Syscall numbers (aarch64):
\\   munmap     = 215
\\   close      = 57
\\   exit       = 93
\\   exit_group = 94
\\
\\ Constants:
\\   BAR0 size = 0x1000000      \\ 16 MB
\\   BAR4 size = 0x2000000000   \\ 128 GB

\\ ------------------------------------------------------------
\\ Raw syscall wrappers
\\ ------------------------------------------------------------

\\ munmap(addr, len) — best effort, return value ignored
munmap addr size :
    ↓ $8 215
    ↓ $0 addr
    ↓ $1 size
    trap

\\ close(fd)
close_fd fd :
    ↓ $8 57
    ↓ $0 fd
    trap

\\ ------------------------------------------------------------
\\ gsp_shutdown — graceful GSP RPC shutdown
\\
\\ Ideal path: send an RPC to GSP asking it to quiesce channels and
\\ park the falcon. For bring-up we skip this; process exit + next
\\ runtime_init's falcon_reset lands the GPU in a known state.
\\ TODO: implement RPC-based shutdown once the host→GSP RPC path is
\\ stable enough to trust during teardown.
\\ ------------------------------------------------------------
gsp_shutdown bar0 :
    \\ no-op for now

\\ ------------------------------------------------------------
\\ runtime_teardown — top-level. Replaces cuCtxDestroy + cuMemFree.
\\
\\ bar0_va : va returned by bar0_map
\\ bar4_va : va returned by bar4_map
\\ vfio_fd : the sysfs resource fd (resource0). A production VFIO
\\           path would also pass resource4's fd; here the caller
\\           closes it via a second close_fd if it kept it around.
\\
\\ Steps:
\\   1. Best-effort GSP shutdown (currently a no-op)
\\   2. munmap BAR0 register window (16 MB)
\\   3. munmap BAR4 HBM window (128 GB)
\\   4. close the sysfs/vfio fd
\\ ------------------------------------------------------------
runtime_teardown bar0_va bar4_va vfio_fd :
    gsp_shutdown bar0_va
    munmap bar0_va 0x1000000
    munmap bar4_va 0x2000000000
    close_fd vfio_fd

\\ ------------------------------------------------------------
\\ runtime_exit — hand control back to the OS
\\
\\ Uses __NR_exit (93) rather than exit_group so a single-threaded
\\ process returns the exact status. For a multi-threaded server
\\ swap to 94 (__NR_exit_group) to tear down all threads atomically.
\\ ------------------------------------------------------------
runtime_exit code :
    ↓ $8 93
    ↓ $0 code
    trap
