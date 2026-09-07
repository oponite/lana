/* Test shared library for the LIP-018 FFI unit test (tests/unit/test_ffi.c).
 *
 * Exports a deterministic arithmetic function (`add`) and a function that
 * deliberately faults (`crash`) so the C VM's crash-containment guard can be
 * exercised. Built as a shared library by CMake; the test loads it via
 * `ffi_load` and calls it via `ffi_call`.
 */

double add(double a, double b) {
    return a + b;
}

/* Deterministic void-returning function (no arguments). */
void noop(void) {
}

/* Deliberately dereference a null pointer to fault. */
void crash(void) {
    volatile int *null_pointer = (volatile int *)0;
    *null_pointer = 42;
}
