if(LANA_BUILD_INTEGRATIONS)
    set_target_properties(lanaruntime PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(lana_bridge SHARED integrations/native/lana_bridge.c)
    target_include_directories(lana_bridge PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/integrations/native/include
        ${LANA_INCLUDE_DIRS})
    target_link_libraries(lana_bridge PRIVATE lanaruntime m)
    target_compile_options(lana_bridge PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_definitions(lana_bridge PRIVATE LANA_VERSION="${LANA_VERSION}")
    set_target_properties(lana_bridge PROPERTIES VERSION ${LANA_VERSION} SOVERSION 1)

    add_executable(lana_bridge_tests integrations/native/test_bridge.c)
    target_link_libraries(lana_bridge_tests PRIVATE lana_bridge)
    target_compile_options(lana_bridge_tests PRIVATE -Wall -Wextra -Wpedantic -Werror -UNDEBUG)
    target_compile_definitions(lana_bridge_tests PRIVATE LANA_VERSION="${LANA_VERSION}")
    add_test(NAME lana_native_bridge
        COMMAND "${CMAKE_COMMAND}"
            -DLANAVM=$<TARGET_FILE:lanavm>
            -DCOMPILER=${LANA_NATIVE_COMPILER}
            -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/integrations/lana/echo_bridge.lana
            -DTEST=$<TARGET_FILE:lana_bridge_tests>
            -DROOT=${CMAKE_CURRENT_BINARY_DIR}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestNativeBridge.cmake")

    add_executable(lana_bridge_pipeline_tests integrations/native/test_bridge_pipeline.c)
    target_link_libraries(lana_bridge_pipeline_tests PRIVATE lana_bridge lanaruntime m)
    target_compile_options(lana_bridge_pipeline_tests PRIVATE -Wall -Wextra -Wpedantic -Werror -UNDEBUG)

    foreach(app IN ITEMS sensor_fusion service_health doc_router advisory_forecast)
        add_test(NAME lana_reference_app_${app}
            COMMAND "${CMAKE_COMMAND}"
                -DLANAVM=$<TARGET_FILE:lanavm>
                -DCOMPILER=${LANA_NATIVE_COMPILER}
                -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/examples/reference-apps/${app}.lana
                -DTEST=$<TARGET_FILE:lana_bridge_pipeline_tests>
                -DROOT=${CMAKE_CURRENT_BINARY_DIR}
                -DFIXTURE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/examples/reference-apps/fixtures/${app}
                -DAPP_NAME=${app}
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestReferenceApp.cmake")
    endforeach()

    install(TARGETS lana_bridge LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(FILES integrations/native/include/lana/bridge.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lana)

    include(cmake/AdapterTargets.cmake)
endif()
