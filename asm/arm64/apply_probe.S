#if defined(__APPLE__)
#define SS_ASM_SYMBOL _ss_asm_apply_probe
#else
#define SS_ASM_SYMBOL ss_asm_apply_probe
#endif

.text
.p2align 2
.globl SS_ASM_SYMBOL
SS_ASM_SYMBOL:
    fcmp d1, #0.0
    b.ge 1f
    fmov d3, #1.0
    fsub d0, d3, d0
    fneg d1, d1
1:
    fsub d3, d0, d2
    fmadd d0, d1, d3, d2
    fmov d3, xzr
    fmov d4, #1.0
    fmax d0, d0, d3
    fmin d0, d0, d4
    ret
