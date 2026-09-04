#include "data.h"
#include "state.h"
#include "vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { (void)fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); return 1; } } while (0)

static int test_state_dist_inspect(void) {
    LanaVM vm;
    LanaStateValue state_a, state_b;
    LanaStateDist *dirac, *left_dirac, *right_dirac, *append, *mapped;
    Value left_value, right_value;
    char *out = NULL;

    lana_vm_init(&vm, NULL);

    /* DIRAC serializes to a single-node JSON document. */
    CHECK(lana_state_make_complex(0.4, 0.0, 0.0, &state_a.state) == LANA_OK);
    CHECK(lana_vm_state_dist_dirac(&vm, &state_a, &dirac) == LANA_OK);
    CHECK(lana_vm_state_dist_inspect(dirac, LANA_INSPECT_JSON, &out) == LANA_OK);
    CHECK(out != NULL);
    CHECK(strcmp(out,
        "{\"kind\":\"state_dist\",\"node_count\":1,\"max_depth\":0,\"append_count\":0,"
        "\"transform_count\":0,\"dirac_count\":1,\"exact_measurement\":true,"
        "\"sampling_required\":false,\"provenance\":{},\"root\":0,\"nodes\":["
        "{\"id\":0,\"kind\":\"dirac\",\"state\":{\"p\":0.4,\"d_re\":0,\"d_im\":0}}]}") == 0);
    free(out);
    out = NULL;

    /* DIRAC also serializes to a DOT document. */
    CHECK(lana_vm_state_dist_inspect(dirac, LANA_INSPECT_DOT, &out) == LANA_OK);
    CHECK(out != NULL);
    CHECK(strncmp(out, "digraph state_dist {\n", 21) == 0);
    CHECK(strstr(out, "n0 [label=\"dirac\\np=0.4\"];") != NULL);
    free(out);
    out = NULL;

    /* APPEND over two Dirac children reports three nodes, depth one. */
    CHECK(lana_state_make_complex(0.2, 0.0, 0.0, &state_a.state) == LANA_OK);
    CHECK(lana_state_make_complex(0.3, 0.0, 0.0, &state_b.state) == LANA_OK);
    CHECK(lana_vm_state_dist_dirac(&vm, &state_a, &left_dirac) == LANA_OK);
    CHECK(lana_vm_state_dist_dirac(&vm, &state_b, &right_dirac) == LANA_OK);
    left_value = lana_value_state_dist(left_dirac);
    right_value = lana_value_state_dist(right_dirac);
    CHECK(lana_vm_state_dist_append(&vm, &left_value, &right_value, &append) == LANA_OK);
    CHECK(lana_vm_state_dist_inspect(append, LANA_INSPECT_JSON, &out) == LANA_OK);
    CHECK(out != NULL);
    CHECK(strstr(out, "\"node_count\":3") != NULL);
    CHECK(strstr(out, "\"max_depth\":1") != NULL);
    CHECK(strstr(out, "\"append_count\":1") != NULL);
    CHECK(strstr(out, "\"dirac_count\":2") != NULL);
    free(out);
    out = NULL;

    /* TRANSFORM over a Dirac child reports a transform node. */
    CHECK(lana_vm_state_dist_transform(&vm, LANA_TRANSFORM_INVERT, dirac, &mapped) == LANA_OK);
    CHECK(lana_vm_state_dist_inspect(mapped, LANA_INSPECT_JSON, &out) == LANA_OK);
    CHECK(out != NULL);
    CHECK(strstr(out, "\"transform_count\":1") != NULL);
    CHECK(strstr(out, "\"kind\":\"transform\",\"transform_id\":0") != NULL);
    free(out);
    out = NULL;

    lana_vm_free(&vm);
    return 0;
}

int main(void) {
    CHECK(test_state_dist_inspect() == 0);
    (void)printf("INSPECT tests passed\n");
    return 0;
}
