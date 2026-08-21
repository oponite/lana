if(NOT DEFINED LANA_SOURCE_DIR OR NOT DEFINED LANA_BUNDLE_OUTPUT)
    message(FATAL_ERROR "LANA_SOURCE_DIR and LANA_BUNDLE_OUTPUT are required")
endif()

set(_lana_modules
    lexer.lana
    syntax.lana
    parser.lana
    resolver.lana
    ir.lana
    emitter.lana
    main.lana
)
set(_lana_bundle "")
foreach(_lana_module IN LISTS _lana_modules)
    file(READ "${LANA_SOURCE_DIR}/compiler/${_lana_module}" _lana_text)
    string(REGEX REPLACE "import[ \t]+\"[^\"]+\"[ \t]+as[ \t]+[A-Za-z_][A-Za-z0-9_]*[ \t]*;[ \t]*\r?\n" "" _lana_text "${_lana_text}")
    foreach(_lana_alias lexer syntax parser resolver ir emitter)
        string(REGEX REPLACE "(^|[^A-Za-z0-9_])${_lana_alias}\\." "\\1" _lana_text "${_lana_text}")
    endforeach()
    string(APPEND _lana_bundle "${_lana_text}\n")
endforeach()
file(WRITE "${LANA_BUNDLE_OUTPUT}" "${_lana_bundle}")
