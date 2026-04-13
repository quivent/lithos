#!/bin/sh
# Run the Lithos compiler via forth-bootstrap.
# Usage: ./run-lithos.sh <input.li> --emit ptx -o output.ptx
#
# The bootstrap's direct-file mode doesn't execute top-level calls properly
# after nested includes, so we launch via a one-line runner.

FORTH=/home/ubuntu/sixth/bootstrap/forth-bootstrap
RUNNER=$(mktemp /tmp/lithos-run-XXXXXX.fs)
echo 's" /home/ubuntu/lithos/compiler/lithos.fs" included' > "$RUNNER"
"$FORTH" "$RUNNER" "$@"
RC=$?
rm -f "$RUNNER"
exit $RC
