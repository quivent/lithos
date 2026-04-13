.ifndef DTC_MACROS_DEFINED
.set DTC_MACROS_DEFINED, 1
.macro NEXT
    ldr     x25, [x26], #8
    ldr     x16, [x25]
    br      x16
.endm
.macro PUSH reg
    str     x22, [x24, #-8]!
    mov     x22, \reg
.endm
.macro POP reg
    mov     \reg, x22
    ldr     x22, [x24], #8
.endm
.macro RPUSH reg
    str     \reg, [x23, #-8]!
.endm
.macro RPOP reg
    ldr     \reg, [x23], #8
.endm
.endif
