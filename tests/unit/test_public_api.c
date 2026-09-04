#include "state.h"
#include "vm.h"

int main(void) {
    LanaState state;
    LanaVM vm;

    if (lana_state_make_complex(0.5, 0.0, 0.0, &state) != LANA_OK) {
        return 1;
    }
    (void)vm;
    return lana_state_valid(&state) ? 0 : 1;
}
