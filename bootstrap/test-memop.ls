main :
    ↓ $8 214
    ↓ $0 0
    trap
    buf = ↑ $0
    ← 8 buf 42
    x = → 8 buf
    ↓ $8 93
    ↓ $0 x
    trap
