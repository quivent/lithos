open path flags mode :
    ↓ $8 56
    ↓ $0 -100
    ↓ $1 path
    ↓ $2 flags
    ↓ $3 mode
    trap
    ↑ $0

main :
    open 0 0 0
    ↓ $8 93
    ↓ $0 0
    trap
