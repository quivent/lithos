# Tokenizer — Design Document

## What the tokenizer is, specifically

Qwen 3.5 27B uses byte-level BPE. The on-disk format is HuggingFace
`tokenizer.json` (13 MB). The runtime need is narrower:

- A vocab map: `bytes → token_id` (248,044 entries + 33 special tokens)
- An ordered merge list: `(token_a, token_b) → rank` (247,587 entries)
- A pre-tokenization regex (GPT-4 style, CJK-aware)
- A byte-level encoding table (256 entries, deterministic)
- NFC unicode normalization on input

Decoding (token_id → bytes) is a flat array lookup. Trivial.

Encoding (bytes → token_ids) is the real work. Per chunk produced by
the pre-tokenizer regex, the BPE merge loop iteratively finds the
lowest-rank adjacent pair and merges. Done.

## What belongs in Lithos and what doesn't

Lithos is a GPU compute language. The compiler emits SASS (when the
bootstrap parses compositions). The runtime loads weights and runs
kernels. The tokenizer is string processing on small inputs on the
CPU. It is orthogonal to everything Lithos is built for.

But the beekeeper's zero-dependency ethos is real: no Python, no
libcuda, no Forth. The tokenizer cannot be a Python subprocess and
cannot link against HuggingFace's Rust `tokenizers` crate. It must
be either:

1. Lithos `.ls` code (pre-tokenization regex is the hard part)
2. ARM64 assembly in the bootstrap layer
3. A standalone C program linked at runtime
4. Precomputed token IDs (hardcoded test prompts, not real use)

This document architects option 2 — bootstrap-level ARM64 — and
defines the interface that the runtime talks to.

## Rationale for ARM64 in the bootstrap

- The bootstrap already has syscall wrappers for `openat`, `mmap`,
  `close`, `write`. The tokenizer needs exactly these.
- String processing in ARM64 is not pretty but it is bounded. The
  tokenizer is ~1,000 instructions, not 100,000.
- The bootstrap is where Lithos loads files before the compiler
  runs. It is the natural home for a file → structured-data step.
- Once written, it never changes. The merge table changes per
  model, but the tokenizer code does not.
- Keeps the ethos: zero external dependencies, zero runtime glue.

## The three parts of the tokenizer

### Part 1: Load (one-time)

Parse `tokenizer.json` once at startup and build a compact in-memory
representation. The JSON is 13 MB and the parser is the painful part.
Output:

```
vocab_table:   248,077 × 8 bytes  = ~2.0 MB   (u64: byte_offset | length)
vocab_bytes:   contiguous blob    = ~3.5 MB   (concatenated token strings)
merge_table:   247,587 × 12 bytes = ~3.0 MB   (u32 a_id, u32 b_id, u32 rank)
merge_hash:    perfect-hash index = ~1.5 MB   (for O(1) rank lookup)
byte_encode:   256 × 4 bytes      = ~1 KB    (byte → initial token_id)
```

Total RAM: ~10 MB, one-time cost.

**Design decision:** parse the JSON in ARM64 assembly directly. This
sounds bad. It is bad. But it's bounded — the format is specific and
the parser only needs to handle the subset HuggingFace emits (no
schema flexibility needed). ~400 lines of assembly.

Alternative: precompile the JSON to a flat binary blob offline, ship
the blob alongside the model. The bootstrap reads the blob in 3
mmap calls and is ready in 5ms. **Prefer this.** Creates a one-time
offline step (runnable from any language) and a permanent zero-cost
runtime path. Blob format:

```
magic:       4 bytes  "LTOK"
version:     4 bytes  u32
vocab_count: 4 bytes  u32
merge_count: 4 bytes  u32
regex_len:   4 bytes  u32
special_cnt: 4 bytes  u32
<vocab_table>  : u64 × vocab_count (offset+length packed)
<vocab_bytes>  : concatenated byte sequences
<merge_table>  : u32 × 3 × merge_count (a_id, b_id, rank)
<regex_pattern>: regex_len bytes (for pre-tokenization)
<special_list> : u32 × special_cnt (token ids that bypass BPE)
```

The offline compiler for this blob is ~150 lines of Python or Rust,
runs once per model, output is deterministic. The runtime loader in
ARM64 is ~80 lines (mmap + pointer arithmetic, no parsing).

### Part 2: Pre-tokenize (per input)

Qwen's regex (from `tokenizer.json`):

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)
| [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
| \p{N}
| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
| \s*[\r\n]+
| \s+(?!\S)
| \s+
```

This is GPT-4's regex plus CJK handling (`\p{L}` covers it). The
regex engine needed is *almost* regular (Unicode categories are a
closed set — precompute a 1 MB table mapping byte → category for
the full BMP, skip to UTF-8 decoding for supplementary planes).

**Design decision:** don't implement a regex engine. Implement this
specific regex as a state machine directly. ~200 lines of ARM64.
The Unicode category table is part of the precomputed blob.

The output of this phase is a list of chunks (byte ranges into the
input). Each chunk is then BPE-encoded independently. Chunks are
typically 1-20 bytes. This bounds the per-chunk BPE cost.

### Part 3: BPE merge loop (per chunk)

For each chunk:
1. Initialize: `tokens[i] = byte_encode[input[i]]` — convert bytes to initial token ids
2. Loop:
   - Scan all adjacent pairs `(tokens[i], tokens[i+1])`
   - Look up rank of each pair in `merge_hash` — if found, record `(rank, i)`
   - Find the lowest-rank pair — if none, done
   - Merge: replace `tokens[i], tokens[i+1]` with the new token id, shift remaining
   - Repeat

Complexity: O(n²) naive, fine because n ≤ 20 per chunk. For a 1000-byte
input that's ~300 chunks × ~20² = 120,000 operations. Microseconds.

**Data structure for `merge_hash`:** the merge table indexes by
`(token_a_id, token_b_id) → rank`. With 247K entries and u32 ids, the
natural hash is CHD (compact hash displacement) or a simple 2-level
table. Or: since token ids are dense in `[0, 248K)`, store as a sparse
array indexed by `(a * 248000 + b) % TABLE_SIZE` with linear probing.
~6 MB table for O(1) lookup. Acceptable.

Alternative: sorted array + binary search. ~3 MB. O(log n) per lookup.
Simpler in assembly. Slower by ~4x. **Prefer for first version.**

~300 lines of ARM64 for the merge loop + lookup.

### Part 4: Special tokens

Special tokens like `<|im_start|>`, `<|im_end|>`, `<|endoftext|>` are
exact-match bypasses. Before pre-tokenization, scan for any special
token byte sequence in the input. If found, split the input at that
boundary, emit the special token id directly, and continue with the
surrounding chunks.

Qwen has 33 special tokens. The scan is a trie or Aho-Corasick. For
33 patterns, a simple linear scan with a trie jump table is fine.
~100 lines.

## Interface to the rest of the system

```
// bootstrap/tokenizer.s
.global tok_load       // (blob_path: ptr) -> status
.global tok_encode     // (input: ptr, len: usize, output: ptr, max_ids: usize) -> n_ids
.global tok_decode     // (ids: ptr, n_ids: usize, output: ptr, max_bytes: usize) -> n_bytes
.global tok_eos_id     // () -> u32
.global tok_bos_id     // () -> u32 (or 0xFFFFFFFF if none)
```

The launcher calls `tok_load("model.ltok")` once at startup. For each
user prompt, calls `tok_encode(prompt, len, buf, 4096)` to get the
token id array. Feeds to the inference loop. When a token id comes
back from sampling, appends to a growing id buffer. On EOS or max
length, calls `tok_decode(ids, n, output_buf, 8192)` to render bytes
and `write` to stdout.

That's the full interface. Five functions. No dependencies.

## Total size estimate

| Component | Lines ARM64 | Notes |
|---|---|---|
| Blob loader | 80 | mmap + pointer arithmetic |
| Unicode category lookup | 40 | table indexed load |
| Pre-tokenization state machine | 200 | Qwen's regex, hand-compiled |
| BPE merge loop | 180 | scan + find-min + merge + shift |
| Binary search on merge table | 50 | sorted array lookup |
| Special token scan | 100 | trie walk |
| Vocab decode | 60 | token_id → bytes |
| Entry points | 40 | tok_load, tok_encode, tok_decode |
| **Total** | **~750** | |

Plus the offline blob compiler: ~150 lines in any host language.

## Testing strategy

The hard part is parity: "The capital of France is" must produce the
same token ids Lithos sees as HuggingFace tokenizers would. Test
approach:

1. Python reference: `AutoTokenizer.from_pretrained(...).encode(prompt)`
   → write golden ids to `tests/tokenizer_golden.txt`
2. Lithos runtime: `tok_encode(prompt)` → compare to golden
3. Fuzz: 1000 random UTF-8 strings from a corpus, compare
4. Edge cases: empty string, pure whitespace, CJK, emoji, control chars,
   very long inputs, special tokens mid-sequence

Golden file lives in `tests/tokenizer_golden.txt`. Regenerated from
Python when the model changes. The bootstrap tokenizer never sees
Python — only the golden ids.

## What this unblocks

- First-token test with real prompts, not hardcoded ids
- Autoregressive generation loop can detect EOS
- Prompt caching can key on byte sequences
- Multi-turn chat becomes possible (the chat template is also just
  a byte sequence that needs tokenizing)

## What this defers

- Streaming decode during generation (decode per-token as ids arrive
  rather than all at once at the end) — trivial extension of `tok_decode`
- Token healing (repair broken UTF-8 boundaries across token outputs)
- Multi-modal tokens (vision, audio) — not needed for text inference
- Custom chat templates — the launcher can format the template as bytes
  before calling `tok_encode`

## Critical path to first real inference token

1. Write the Python blob compiler (~1 hour)
2. Compile `tokenizer.json` → `model.ltok` (~5 seconds runtime)
3. Write `tok_load` + `tok_decode` in ARM64 (~1 day)
4. Test decode parity against Python golden
5. Write pre-tokenization state machine (~2 days, this is the hard part)
6. Write BPE merge loop + binary search (~1 day)
7. Write special token scan (~half day)
8. Encode parity test against Python golden
9. Wire `tok_encode` into the launcher before the forward pass
10. Wire `tok_decode` into the launcher after sampling

~5-7 days of focused work. Parallelizable if the blob compiler and
the ARM64 pieces are developed against the same golden tests.

## Decision points for the beekeeper

1. **Offline blob vs runtime JSON parse?** Document recommends blob.
   Runtime parse adds ~400 lines of JSON parsing in ARM64 that nobody
   wants to write or debug.

2. **Regex state machine vs regex engine?** Document recommends
   hand-compiling Qwen's specific regex as a state machine. A general
   regex engine is 2000+ lines in ARM64 and only this one pattern is
   ever used. Hand-compiling sacrifices reusability for ~10x smaller
   code.

3. **Binary search vs hash table on merge lookup?** Document recommends
   binary search for v1 (simpler, ~3 MB). Hash table is a v2 perf
   improvement if needed.

4. **Where does the blob live?** Next to the model weights is natural:
   `/home/ubuntu/models/.../model.ltok`. Alternative: bundle in the
   Lithos repo. First option scales to multiple models, second makes
   Lithos more self-contained. Beekeeper's call.

5. **Special tokens hardcoded or data-driven?** Data-driven means the
   blob format carries the 33 special token ids. Hardcoded means each
   model needs a recompile. Data-driven wins, negligible cost.

---

This document is design, not commitment. The tokenizer is a real
gap (M1 in PLAN.md) and ~750 lines of ARM64 is a real investment.
Alternative: stub it with hardcoded test prompt ids, ship first-token,
come back to it when interactive use becomes the actual need. The
beekeeper decides when this transitions from architect to build.
