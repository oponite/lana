#include "ss/asm_probe.h"

double ss_asm_apply_probe(double source_p, double source_d, double target_p) {
    double influence_target = source_d < 0.0 ? 1.0 - source_p : source_p;
    double strength = source_d < 0.0 ? -source_d : source_d;
    double result = target_p + strength * (influence_target - target_p);
    if (result < 0.0) return 0.0;
    if (result > 1.0) return 1.0;
    return result;
}
