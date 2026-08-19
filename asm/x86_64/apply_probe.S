#if defined(__APPLE__)
#define SS_ASM_SYMBOL _ss_asm_apply_probe
#else
#define SS_ASM_SYMBOL ss_asm_apply_probe
#endif

.text
.globl SS_ASM_SYMBOL
SS_ASM_SYMBOL:
    xorpd %xmm3, %xmm3
    ucomisd %xmm3, %xmm1
    jae 1f
    movsd one_value(%rip), %xmm3
    subsd %xmm0, %xmm3
    movapd %xmm3, %xmm0
    mulsd minus_one_value(%rip), %xmm1
1:
    subsd %xmm2, %xmm0
    mulsd %xmm1, %xmm0
    addsd %xmm2, %xmm0
    maxsd zero_value(%rip), %xmm0
    minsd one_value(%rip), %xmm0
    ret

#if defined(__APPLE__)
.const_data
#else
.section .rodata
#endif
.p2align 3
zero_value:
    .double 0.0
one_value:
    .double 1.0
minus_one_value:
    .double -1.0
