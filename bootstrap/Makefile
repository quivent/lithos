# Makefile — Lithos bootstrap.
#
# The real assembly+link logic lives in build.sh (it has to patch
# sources at build time to resolve the _start collision and promote
# cross-file symbols to .global). This Makefile is a thin wrapper.

BIN      := lithos-bootstrap
BUILDDIR := build
BUILD_SH := ./build.sh

.PHONY: all clean test rebuild

all: $(BIN)

$(BIN): $(BUILD_SH) $(wildcard *.s)
	$(BUILD_SH)

rebuild: clean all

clean:
	rm -rf $(BUILDDIR) $(BIN) *.o

# Sanity check: `1 2 + . bye` over stdin must print "3".
# Output is not followed by newline, so match on the raw bytes.
test: all
	@out=$$(echo '1 2 + . bye' | ./$(BIN)); \
	case "$$out" in \
	  3*) echo "PASS: 1 2 + . bye -> $$out" ;; \
	  *)  echo "FAIL: expected '3', got '$$out'"; exit 1 ;; \
	esac
	@out=$$(echo '10 20 * . bye' | ./$(BIN)); \
	case "$$out" in \
	  c8*) echo "PASS: 10 20 * . bye -> $$out (base 16)" ;; \
	  200*) echo "PASS: 10 20 * . bye -> $$out (base 10)" ;; \
	  *)   echo "FAIL: expected c8 or 200, got '$$out'"; exit 1 ;; \
	esac
