# Differential check for Phase-5 optimization: compile the same source with and
# without `--optimize`, assemble both, run both, and require byte-identical
# stdout and exit status. The unoptimized emitter is the reference; the optimizer
# must not change observable behavior (return values, stdout, effects).
if(NOT DEFINED LANA_VM OR NOT DEFINED LANA_COMPILER OR NOT DEFINED LANA_INPUT
   OR NOT DEFINED LANA_OUTPUT)
    message(FATAL_ERROR "optimization differential test paths are required")
endif()

set(_unopt_lasm "${LANA_OUTPUT}.unopt.lasm")
set(_opt_lasm   "${LANA_OUTPUT}.opt.lasm")
set(_unopt_labc "${LANA_OUTPUT}.unopt.labc")
set(_opt_labc   "${LANA_OUTPUT}.opt.labc")

execute_process(
    COMMAND "${LANA_VM}" run "${LANA_COMPILER}"
            --memory-limit-mib 256 --instruction-limit 100000000
            -- "${LANA_INPUT}" "${_unopt_lasm}"
    RESULT_VARIABLE _unopt_compile)
if(NOT _unopt_compile EQUAL 0)
    message(FATAL_ERROR "unoptimized compile failed: ${_unopt_compile}")
endif()

execute_process(
    COMMAND "${LANA_VM}" run "${LANA_COMPILER}"
            --memory-limit-mib 256 --instruction-limit 100000000
            -- --optimize "${LANA_INPUT}" "${_opt_lasm}"
    RESULT_VARIABLE _opt_compile)
if(NOT _opt_compile EQUAL 0)
    message(FATAL_ERROR "optimized compile failed: ${_opt_compile}")
endif()

execute_process(COMMAND "${LANA_VM}" asm "${_unopt_lasm}" -o "${_unopt_labc}"
                RESULT_VARIABLE _unopt_asm)
execute_process(COMMAND "${LANA_VM}" asm "${_opt_lasm}" -o "${_opt_labc}"
                RESULT_VARIABLE _opt_asm)
if(NOT _unopt_asm EQUAL 0 OR NOT _opt_asm EQUAL 0)
    message(FATAL_ERROR "assemble failed: unopt=${_unopt_asm} opt=${_opt_asm}")
endif()

execute_process(COMMAND "${LANA_VM}" run "${_unopt_labc}"
                OUTPUT_VARIABLE _unopt_out ERROR_VARIABLE _unopt_err
                RESULT_VARIABLE _unopt_run)
execute_process(COMMAND "${LANA_VM}" run "${_opt_labc}"
                OUTPUT_VARIABLE _opt_out ERROR_VARIABLE _opt_err
                RESULT_VARIABLE _opt_run)

if(NOT _unopt_run EQUAL _opt_run)
    message(FATAL_ERROR "exit status diverged: unopt=${_unopt_run} opt=${_opt_run}")
endif()
if(NOT "${_unopt_out}" STREQUAL "${_opt_out}")
    message(FATAL_ERROR "stdout diverged:\n-- unoptimized --\n${_unopt_out}\n-- optimized --\n${_opt_out}")
endif()
if(NOT "${_unopt_err}" STREQUAL "${_opt_err}")
    message(FATAL_ERROR "stderr diverged:\n-- unoptimized --\n${_unopt_err}\n-- optimized --\n${_opt_err}")
endif()

file(REMOVE "${_unopt_lasm}" "${_opt_lasm}" "${_unopt_labc}" "${_opt_labc}")
