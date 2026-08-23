# 设置CMake的最低版本要求
cmake_minimum_required(VERSION 3.10)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 定义项目名称
project(npusim_project LANGUAGES CXX)

include(CTest)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 从环境变量读取 SYSTEMC_HOME
if(NOT DEFINED ENV{SYSTEMC_HOME})
    message(FATAL_ERROR "The environment variable SYSTEMC_HOME is not set. Please export SYSTEMC_HOME before running CMake.")
endif()

set(SYSTEMC_HOME $ENV{SYSTEMC_HOME})
message(STATUS "SystemC Home: ${SYSTEMC_HOME}")
# set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes -DBROAD_W=16")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -DBROAD_W=16")

# SystemC库路径
link_directories(${SYSTEMC_HOME}/lib-linux64)

# 包含目录
include_directories(
    ${SYSTEMC_HOME}/include
    ./llm/include
)


option(BUILD_RUNTIME_SANITIZER_TESTS
    "Build the representative npusim ASan/UBSan/LSan runtime gate" OFF)
option(SYSTEMC_PTHREAD_COROUTINES
    "Assert that SYSTEMC_HOME was built with ENABLE_PTHREADS=ON" OFF)
if(BUILD_RUNTIME_SANITIZER_TESTS)
    if(NOT SYSTEMC_PTHREAD_COROUTINES)
        message(FATAL_ERROR
            "BUILD_RUNTIME_SANITIZER_TESTS requires a SystemC build with "
            "ENABLE_PTHREADS=ON and -DSYSTEMC_PTHREAD_COROUTINES=ON; "
            "QuickThreads uses unregistered fiber stacks and yields false "
            "ASan stack-buffer-overflow reports.")
    endif()
    # This must precede add_subdirectory(DRAMSys) so every SystemC client uses
    # the coroutine ABI selected by the external pthread SystemC library.
    add_compile_definitions(SC_USE_PTHREADS)
endif()


add_subdirectory(DRAMSys)

# 搜索所有.cpp文件
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS ./llm/*.cpp)
list(FILTER SRC_FILES EXCLUDE REGEX ".*/unittest/.*")
# Remove 'test_dramsys.cpp' from the list of sources
list(FILTER SRC_FILES EXCLUDE REGEX ".*/test_dramsys\\.cpp$")
list(FILTER SRC_FILES EXCLUDE REGEX ".*/main\\.cpp")
list(FILTER SRC_FILES EXCLUDE REGEX ".*/program_finalizer_main\\.cpp$")
list(FILTER SRC_FILES EXCLUDE REGEX ".*/build/.*")
list(FILTER SRC_FILES EXCLUDE REGEX ".*/test/.*")
list(FILTER SRC_FILES EXCLUDE REGEX ".*/config_generation/.*")


# # 定义生成的目标
message(STATUS "Source files: ${SRC_FILES}")

# find_package(Cairo REQUIRED)
# include_directories(${CAIRO_INCLUDE_DIRS})

option(BUILD_DEBUG_TARGETS "Build debug versions of executables" OFF)
option(BUILD_SANITIZER_TESTS
    "Build the focused Program Format ASan/UBSan/LSan gate" OFF)
option(BUILD_N7_NIGHTLY_TESTS
    "Register the multi-minute N7 TP4/L2 production timing gate" OFF)
if(BUILD_RUNTIME_SANITIZER_TESTS AND NOT BUILD_SANITIZER_TESTS)
    message(FATAL_ERROR
        "BUILD_RUNTIME_SANITIZER_TESTS requires BUILD_SANITIZER_TESTS=ON")
endif()
# 设置 L1/L2 缓存大小（支持命令行传入，带默认值）
set(L1CACHESIZE ${L1CACHESIZE} CACHE STRING "L1 Cache Size in bytes")
set(L2CACHESIZE ${L2CACHESIZE} CACHE STRING "L2 Cache Size in bytes")

# 设置默认值（如果用户没传）
if(NOT L1CACHESIZE)
    set(L1CACHESIZE 4194304)
endif()

if(NOT L2CACHESIZE)
    set(L2CACHESIZE 15099494)
endif()

# 输出用于调试
message(STATUS "L1CACHESIZE: ${L1CACHESIZE}")
message(STATUS "L2CACHESIZE: ${L2CACHESIZE}")

function(add_test_executable target_name test_main_file)
    add_executable(${target_name} ${SRC_FILES} ${test_main_file})
    target_link_libraries(${target_name}
        PUBLIC
            systemc 
            cairo 
            sfml-graphics 
            sfml-system 
            sfml-window 
            m 
            pthread
            DRAMSys::libdramsys
            DRAMSys::util
            DRAMSys::config
    )
    target_compile_options(${target_name} PRIVATE 
        -Ofast 
        -Wno-unused-result 
        -Wno-ignored-pragmas 
        -Wno-unknown-attributes 
        -DBROAD_W=16
    )
    # 传递 L1CACHESIZE 和 L2CACHESIZE 到编译器
    target_compile_definitions(${target_name} PRIVATE
        L1CACHESIZE=${L1CACHESIZE}
        L2CACHESIZE=${L2CACHESIZE}
        NPUSIM_SOURCE_ROOT="${CMAKE_SOURCE_DIR}"
    )

    
    # 如果启用了调试版本构建选项，则添加调试版本
    if(BUILD_DEBUG_TARGETS)
        add_executable(${target_name}_debug ${SRC_FILES} ${test_main_file})
        target_link_libraries(${target_name}_debug
            PUBLIC
                systemc 
                cairo 
                sfml-graphics 
                sfml-system 
                sfml-window 
                m 
                pthread
                DRAMSys::libdramsys
                DRAMSys::util
                DRAMSys::config
        )
        target_compile_options(${target_name}_debug PRIVATE 
            -g 
            -O0 
            -Wno-unused-result 
            -Wno-ignored-pragmas 
            -Wno-unknown-attributes  
            -DBROAD_W=16
        )
        target_compile_definitions(${target_name}_debug PRIVATE
            L1CACHESIZE=${L1CACHESIZE}
            L2CACHESIZE=${L2CACHESIZE}
            NPUSIM_SOURCE_ROOT="${CMAKE_SOURCE_DIR}"
        )
    endif()
endfunction()

add_test_executable(npusim       "./llm/unittest/npusim.cpp")
add_executable(npusim_program_io_selftest
    ./llm/unittest/program_io_selftest_main.cpp
    ./llm/src/frontend/program_io.cpp
    ./llm/src/frontend/program_finalizer.cpp
    ./llm/src/isa/opcode.cpp
    ./llm/src/isa/record_codec.cpp
    ./llm/src/isa/program_format.cpp
    ./llm/src/memory/hbm_address_map.cpp
    ./llm/src/memory/hbm_runtime.cpp
    ./llm/src/memory/sram_access_unit.cpp
    ./llm/src/memory/sram_region.cpp
    ./llm/src/memory/sram_storage.cpp)
target_link_libraries(npusim_program_io_selftest PRIVATE
    systemc nlohmann_json::nlohmann_json m pthread
    DRAMSys::libdramsys
    DRAMSys::util
    DRAMSys::config)
target_compile_options(npusim_program_io_selftest PRIVATE
    -Wall -Wextra -Werror -ffunction-sections -fdata-sections)
target_link_options(npusim_program_io_selftest PRIVATE
    -Wl,--gc-sections)

if(BUILD_RUNTIME_SANITIZER_TESTS)
    # Rebuild the production simulator sources with instrumentation.  The
    # P8 runtime CTest below executes the same artifact/probe/trace runner as
    # the normal gate, rather than substituting a parser-only harness.
    add_test_executable(npusim_runtime_sanitizer
        "./llm/unittest/npusim.cpp")
    target_compile_options(npusim_runtime_sanitizer PRIVATE
        -O1 -fno-omit-frame-pointer -fsanitize=address,undefined)
    target_link_libraries(npusim_runtime_sanitizer PRIVATE
        -fsanitize=address,undefined)
endif()

# Small compiler-backend reference utility. It links only the public ISA
# container/record codecs, so producing an artifact does not initialize
# SystemC or simulator-global state.
add_executable(npusim_program_fixture
    ./llm/unittest/program_fixture.cpp
    ./llm/src/isa/opcode.cpp
    ./llm/src/isa/record_codec.cpp
    ./llm/src/isa/program_format.cpp)
target_compile_options(npusim_program_fixture PRIVATE
    -Wall -Wextra -Werror)

# Strict Python-manifest to ProgramArtifact finalization gate.  This remains a
# lightweight compiler-backend target: JSON + public ISA codecs only, with no
# SystemC initialization or simulator-global state.
add_executable(npusim_program_finalizer_selftest
    ./llm/unittest/program_finalizer_selftest_main.cpp
    ./llm/src/frontend/program_finalizer.cpp
    ./llm/src/isa/opcode.cpp
    ./llm/src/isa/record_codec.cpp
    ./llm/src/isa/program_format.cpp)
target_link_libraries(npusim_program_finalizer_selftest PRIVATE
    nlohmann_json::nlohmann_json)
target_compile_options(npusim_program_finalizer_selftest PRIVATE
    -Wall -Wextra -Werror)

add_executable(npusim_program_finalizer
    ./llm/src/frontend/program_finalizer_main.cpp
    ./llm/src/frontend/program_finalizer.cpp
    ./llm/src/isa/opcode.cpp
    ./llm/src/isa/record_codec.cpp
    ./llm/src/isa/program_format.cpp)
target_link_libraries(npusim_program_finalizer PRIVATE
    nlohmann_json::nlohmann_json)
target_compile_options(npusim_program_finalizer PRIVATE
    -Wall -Wextra -Werror)

if(BUILD_SANITIZER_TESTS)
    add_executable(npusim_program_format_sanitizer_selftest
        ./llm/unittest/program_format_selftest_main.cpp
        ./llm/src/isa/opcode.cpp
        ./llm/src/isa/record_codec.cpp
        ./llm/src/isa/program_format.cpp
        ./llm/src/isa/program_format_selftest.cpp)
    target_compile_options(npusim_program_format_sanitizer_selftest PRIVATE
        -Wall -Wextra -Werror -O1 -fno-omit-frame-pointer
        -fsanitize=address,undefined)
    target_link_libraries(npusim_program_format_sanitizer_selftest PRIVATE
        -fsanitize=address,undefined)
endif()

if(BUILD_TESTING)
    add_test(NAME program_io_v1_selftest
        COMMAND npusim_program_io_selftest)
    set_tests_properties(program_io_v1_selftest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        RUN_SERIAL TRUE
        LABELS "frontend;program;n6;program-io;negative")

    add_test(NAME program_io_cli_preflight
        COMMAND ${CMAKE_COMMAND}
            -DNPUSIM=$<TARGET_FILE:npusim>
            -P ${CMAKE_SOURCE_DIR}/llm/test/frontend/integration/check_program_io_cli_preflight.cmake)
    set_tests_properties(program_io_cli_preflight PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        RUN_SERIAL TRUE
        LABELS "frontend;program;n6;program-io;cli;negative")

    add_test(NAME program_finalizer_v1_selftest
        COMMAND npusim_program_finalizer_selftest)
    set_tests_properties(program_finalizer_v1_selftest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        LABELS "frontend;isa;program;n6;negative")

    if(TARGET npusim_program_format_sanitizer_selftest)
        add_test(NAME program_format_v1_sanitizer_selftest
            COMMAND npusim_program_format_sanitizer_selftest)
        set_tests_properties(program_format_v1_sanitizer_selftest PROPERTIES
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            RUN_SERIAL TRUE
            LABELS "isa;program;p8;negative;sanitizer"
            ENVIRONMENT
                "ASAN_OPTIONS=detect_leaks=1:strict_string_checks=1:halt_on_error=1;UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1")
    endif()

    add_test(NAME isa_v1_selftest
             COMMAND npusim --isa-v1-selftest)
    set_tests_properties(isa_v1_selftest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        RUN_SERIAL TRUE
        LABELS "isa")

    add_test(NAME collective_data_v1_prim_selftest
             COMMAND npusim --collective-data-v1-prim-selftest)
    set_tests_properties(collective_data_v1_prim_selftest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        RUN_SERIAL TRUE
        LABELS "isa")

    add_test(NAME collective_phase_barrier_v1_prim_selftest
             COMMAND npusim --collective-phase-barrier-v1-prim-selftest)
    set_tests_properties(collective_phase_barrier_v1_prim_selftest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        RUN_SERIAL TRUE
        LABELS "isa")

    add_test(NAME sync_runtime_selftest
             COMMAND npusim --sync-runtime-selftest)
    set_tests_properties(sync_runtime_selftest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        RUN_SERIAL TRUE
        LABELS "isa;runtime")

    add_test(NAME p2p_payload_selftest
             COMMAND npusim --p2p-payload-selftest)
    set_tests_properties(p2p_payload_selftest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        RUN_SERIAL TRUE
        LABELS "isa;runtime;p2p")

    add_test(NAME p2p_session_selftest
             COMMAND npusim --p2p-session-selftest)
    set_tests_properties(p2p_session_selftest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        RUN_SERIAL TRUE
        LABELS "isa;runtime;p2p")

    add_test(NAME p5_memory_probe_selftest
             COMMAND npusim --p5-memory-probe-selftest)
    set_tests_properties(p5_memory_probe_selftest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        RUN_SERIAL TRUE
        LABELS "isa;program;p2p;memory-probe")

    # Program runtime tests write traces, event logs, and result files. Keep
    # them below one dedicated binary-tree directory. The compatibility links
    # preserve the simulator existing ../font and ../DRAMSys lookup contract.
    set(program_runtime_dir "${CMAKE_BINARY_DIR}/program-runtime-ctest")
    file(MAKE_DIRECTORY "${program_runtime_dir}")
    if(NOT EXISTS "${CMAKE_BINARY_DIR}/font")
        execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
            "${CMAKE_SOURCE_DIR}/font" "${CMAKE_BINARY_DIR}/font")
    endif()
    if(NOT EXISTS "${CMAKE_BINARY_DIR}/DRAMSys/configs")
        execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
            "${CMAKE_SOURCE_DIR}/DRAMSys/configs"
            "${CMAKE_BINARY_DIR}/DRAMSys/configs")
    endif()

    set(program_fixture_path
        "${program_runtime_dir}/program fixture 路径.npup")
    add_test(NAME program_fixture_generate
             COMMAND npusim_program_fixture "${program_fixture_path}")
    set_tests_properties(program_fixture_generate PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        FIXTURES_SETUP program_artifact
        LABELS "isa;program;sram-bind")

    add_test(NAME program_cli_smoke
             COMMAND npusim
                --program "${program_fixture_path}"
                --hardware-config ${CMAKE_SOURCE_DIR}/llm/test/default/hardware.json
                --simulation-config ${CMAKE_SOURCE_DIR}/llm/test/default/simulation.json
                --mapping-config ${CMAKE_SOURCE_DIR}/llm/test/default/mapping.spec
                --trace-window 1000000)
    set_tests_properties(program_cli_smoke PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        FIXTURES_REQUIRED program_artifact
        TIMEOUT 60
        RUN_SERIAL TRUE
        LABELS "isa;program;startup;runtime;sram-bind"
        PASS_REGULAR_EXPRESSION "Core 0 end compute primitive Dummy_p"
        FAIL_REGULAR_EXPRESSION "\\[PROTO_WAIT\\]")

    add_test(NAME program_cli_mutual_exclusion
             COMMAND npusim
                --program "${program_fixture_path}"
                --workload-config ${CMAKE_SOURCE_DIR}/llm/test/default/workload.json)
    set_tests_properties(program_cli_mutual_exclusion PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        FIXTURES_REQUIRED program_artifact
        WILL_FAIL TRUE
        LABELS "isa;program;negative")

    add_test(NAME program_cli_missing_file
             COMMAND npusim --program
                "${program_runtime_dir}/missing program artifact.npup")
    set_tests_properties(program_cli_missing_file PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        WILL_FAIL TRUE
        LABELS "isa;program;negative")

    set(program_corrupt_fixture_path
        "${program_runtime_dir}/corrupt program artifact.npup")
    add_test(NAME program_corrupt_fixture_generate
             COMMAND npusim_program_fixture
                "${program_corrupt_fixture_path}" --corrupt-crc)
    set_tests_properties(program_corrupt_fixture_generate PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        FIXTURES_SETUP corrupt_program_artifact
        LABELS "isa;program;negative")

    add_test(NAME program_cli_corrupt_crc
             COMMAND npusim --program "${program_corrupt_fixture_path}")
    set_tests_properties(program_cli_corrupt_crc PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        FIXTURES_REQUIRED corrupt_program_artifact
        WILL_FAIL TRUE
        LABELS "isa;program;negative")

    set(program_missing_bind_fixture_path
        "${program_runtime_dir}/missing bind program.npup")
    add_test(NAME program_missing_bind_fixture_generate
             COMMAND npusim_program_fixture
                "${program_missing_bind_fixture_path}" --missing-bind)
    set_tests_properties(program_missing_bind_fixture_generate PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        FIXTURES_SETUP missing_bind_program_artifact
        LABELS "isa;program;negative;sram-bind")

    add_test(NAME program_cli_missing_bind
             COMMAND npusim
                --program "${program_missing_bind_fixture_path}"
                --hardware-config ${CMAKE_SOURCE_DIR}/llm/test/default/hardware.json
                --simulation-config ${CMAKE_SOURCE_DIR}/llm/test/default/simulation.json
                --mapping-config ${CMAKE_SOURCE_DIR}/llm/test/default/mapping.spec)
    set_tests_properties(program_cli_missing_bind PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        FIXTURES_REQUIRED missing_bind_program_artifact
        WILL_FAIL TRUE
        TIMEOUT 60
        RUN_SERIAL TRUE
        LABELS "isa;program;negative;sram-bind"
        FAIL_REGULAR_EXPRESSION "Loaded Program Format")

    set(program_wrong_arity_fixture_path
        "${program_runtime_dir}/wrong arity program.npup")
    add_test(NAME program_wrong_arity_fixture_generate
             COMMAND npusim_program_fixture
                "${program_wrong_arity_fixture_path}" --wrong-arity)
    set_tests_properties(program_wrong_arity_fixture_generate PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        FIXTURES_SETUP wrong_arity_program_artifact
        LABELS "isa;program;negative;sram-bind")

    add_test(NAME program_cli_wrong_bind_arity
             COMMAND npusim
                --program "${program_wrong_arity_fixture_path}"
                --hardware-config ${CMAKE_SOURCE_DIR}/llm/test/default/hardware.json
                --simulation-config ${CMAKE_SOURCE_DIR}/llm/test/default/simulation.json
                --mapping-config ${CMAKE_SOURCE_DIR}/llm/test/default/mapping.spec)
    set_tests_properties(program_cli_wrong_bind_arity PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        FIXTURES_REQUIRED wrong_arity_program_artifact
        WILL_FAIL TRUE
        TIMEOUT 60
        RUN_SERIAL TRUE
        LABELS "isa;program;negative;sram-bind"
        FAIL_REGULAR_EXPRESSION "Loaded Program Format")

    set(program_sync_event_fixture_path
        "${program_runtime_dir}/sync event program.npup")
    add_test(NAME program_sync_event_fixture_generate
             COMMAND npusim_program_fixture
                "${program_sync_event_fixture_path}" --p4-sync-event)
    set_tests_properties(program_sync_event_fixture_generate PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        FIXTURES_SETUP sync_event_program_artifact
        LABELS "isa;program;sync;event")

    add_test(NAME program_cli_sync_event
             COMMAND npusim
                --program "${program_sync_event_fixture_path}"
                --hardware-config ${CMAKE_SOURCE_DIR}/llm/test/default/hardware.json
                --simulation-config ${CMAKE_SOURCE_DIR}/llm/test/default/simulation.json
                --mapping-config ${CMAKE_SOURCE_DIR}/llm/test/default/mapping.spec
                --trace-window 1000000)
    set_tests_properties(program_cli_sync_event PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        FIXTURES_REQUIRED sync_event_program_artifact
        TIMEOUT 60
        RUN_SERIAL TRUE
        LABELS "isa;program;runtime;sync;event"
        PASS_REGULAR_EXPRESSION
            "\\[COLL_DRAIN\\].*barriers=0.*event=0"
        FAIL_REGULAR_EXPRESSION "\\[PROTO_WAIT\\]")

    set(program_sync_bad_endpoint_fixture_path
        "${program_runtime_dir}/sync bad endpoint program.npup")
    add_test(NAME program_sync_bad_endpoint_fixture_generate
             COMMAND npusim_program_fixture
                "${program_sync_bad_endpoint_fixture_path}"
                --p4-sync-bad-endpoint)
    set_tests_properties(program_sync_bad_endpoint_fixture_generate PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        FIXTURES_SETUP sync_bad_endpoint_program_artifact
        LABELS "isa;program;negative;sync;event")

    add_test(NAME program_cli_sync_bad_endpoint
             COMMAND npusim
                --program "${program_sync_bad_endpoint_fixture_path}"
                --hardware-config ${CMAKE_SOURCE_DIR}/llm/test/default/hardware.json
                --simulation-config ${CMAKE_SOURCE_DIR}/llm/test/default/simulation.json
                --mapping-config ${CMAKE_SOURCE_DIR}/llm/test/default/mapping.spec)
    set_tests_properties(program_cli_sync_bad_endpoint PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        FIXTURES_REQUIRED sync_bad_endpoint_program_artifact
        WILL_FAIL TRUE
        TIMEOUT 60
        RUN_SERIAL TRUE
        LABELS "isa;program;negative;sync;event"
        FAIL_REGULAR_EXPRESSION "Loaded Program Format")

    set(program_lifecycle_fixture_path
        "${program_runtime_dir}/sram lifecycle program.npup")
    add_test(NAME program_lifecycle_fixture_generate
             COMMAND npusim_program_fixture
                "${program_lifecycle_fixture_path}" --p4-lifecycle)
    set_tests_properties(program_lifecycle_fixture_generate PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        FIXTURES_SETUP lifecycle_program_artifact
        LABELS "isa;program;sram;lifecycle")

    add_test(NAME program_cli_sram_lifecycle
             COMMAND npusim
                --program "${program_lifecycle_fixture_path}"
                --hardware-config ${CMAKE_SOURCE_DIR}/llm/test/sram/hardware.json
                --simulation-config ${CMAKE_SOURCE_DIR}/llm/test/sram/simulation.json
                --mapping-config ${CMAKE_SOURCE_DIR}/llm/test/default/mapping.spec
                --trace-window 1000000)
    set_tests_properties(program_cli_sram_lifecycle PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        FIXTURES_REQUIRED lifecycle_program_artifact
        TIMEOUT 60
        RUN_SERIAL TRUE
        LABELS "isa;program;runtime;sram;lifecycle"
        PASS_REGULAR_EXPRESSION "End DONE reception"
        FAIL_REGULAR_EXPRESSION "\\[PROTO_WAIT\\]")

    set(program_lifecycle_dangling_fixture_path
        "${program_runtime_dir}/sram lifecycle dangling program.npup")
    add_test(NAME program_lifecycle_dangling_fixture_generate
             COMMAND npusim_program_fixture
                "${program_lifecycle_dangling_fixture_path}"
                --p4-lifecycle-dangling)
    set_tests_properties(program_lifecycle_dangling_fixture_generate PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        FIXTURES_SETUP lifecycle_dangling_program_artifact
        LABELS "isa;program;negative;sram;lifecycle")

    add_test(NAME program_cli_sram_lifecycle_dangling
             COMMAND npusim
                --program "${program_lifecycle_dangling_fixture_path}"
                --hardware-config ${CMAKE_SOURCE_DIR}/llm/test/sram/hardware.json
                --simulation-config ${CMAKE_SOURCE_DIR}/llm/test/sram/simulation.json
                --mapping-config ${CMAKE_SOURCE_DIR}/llm/test/default/mapping.spec)
    set_tests_properties(program_cli_sram_lifecycle_dangling PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        FIXTURES_REQUIRED lifecycle_dangling_program_artifact
        WILL_FAIL TRUE
        TIMEOUT 60
        RUN_SERIAL TRUE
        LABELS "isa;program;negative;sram;lifecycle"
        FAIL_REGULAR_EXPRESSION "Loaded Program Format")

    find_package(PythonInterp 3 REQUIRED)

    set(program_io_e1_t_runner
        "${CMAKE_SOURCE_DIR}/llm/test/frontend/integration/run_program_io_e1_t.py")
    add_test(NAME program_io_e1_t_runtime
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_SOURCE_DIR}:${CMAKE_SOURCE_DIR}/llm/test/frontend/unit
            ${PYTHON_EXECUTABLE} -B ${program_io_e1_t_runner}
            --npusim $<TARGET_FILE:npusim>
            --finalizer $<TARGET_FILE:npusim_program_finalizer>
            --resolver $<TARGET_FILE:npusim_program_io_selftest>
            --hardware ${CMAKE_SOURCE_DIR}/llm/test/sram/hardware_numa.json
            --simulation ${CMAKE_SOURCE_DIR}/llm/test/sram/simulation.json
            --mapping ${CMAKE_SOURCE_DIR}/llm/test/default/mapping.spec
            --runtime-root ${program_runtime_dir})
    set_tests_properties(program_io_e1_t_runtime PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        TIMEOUT 300
        RUN_SERIAL TRUE
        LABELS "frontend;program;n6;program-io;e1;runtime;interop"
        PASS_REGULAR_EXPRESSION "\\[PROGRAM_IO E1-T\\] PASS"
        FAIL_REGULAR_EXPRESSION "\\[PROTO_WAIT\\]")

    set(swizzle_runtime_runner
        "${CMAKE_SOURCE_DIR}/llm/test/frontend/integration/run_swizzle_runtime.py")
    add_test(NAME swizzle_runtime_comparison
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_SOURCE_DIR}:${CMAKE_SOURCE_DIR}/llm/test/frontend/integration
            ${PYTHON_EXECUTABLE} -B ${swizzle_runtime_runner}
            --mode suite
            --npusim $<TARGET_FILE:npusim>
            --finalizer $<TARGET_FILE:npusim_program_finalizer>
            --resolver $<TARGET_FILE:npusim_program_io_selftest>
            --hardware ${CMAKE_SOURCE_DIR}/notes/frontend/examples/hardware_2x1.json
            --simulation ${CMAKE_SOURCE_DIR}/llm/test/sram/simulation.json
            --mapping ${CMAKE_SOURCE_DIR}/llm/test/default/mapping.spec
            --runtime-root ${program_runtime_dir}
            --report ${program_runtime_dir}/swizzle-runtime-comparison-report.json
            --timeout 600)
    set_tests_properties(swizzle_runtime_comparison PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        TIMEOUT 1200
        RUN_SERIAL TRUE
        LABELS "frontend;program;swizzle;runtime;comparison;interop"
        PASS_REGULAR_EXPRESSION
            "\\[SWIZZLE RUNTIME COMPARISON\\] PASS: timing_execution=1 functional_execution=0 cases=3 branches=6 repeat=2"
        FAIL_REGULAR_EXPRESSION
            "\\[PROTO_WAIT\\]|Swizzle runtime first failure|Traceback")

    set(swizzle_scale_runtime_runner
        "${CMAKE_SOURCE_DIR}/llm/test/frontend/integration/run_swizzle_scale_runtime.py")
    add_test(NAME swizzle_scale_runtime_comparison
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_SOURCE_DIR}:${CMAKE_SOURCE_DIR}/llm/test/frontend/integration
            ${PYTHON_EXECUTABLE} -B ${swizzle_scale_runtime_runner}
            --npusim $<TARGET_FILE:npusim>
            --finalizer $<TARGET_FILE:npusim_program_finalizer>
            --resolver $<TARGET_FILE:npusim_program_io_selftest>
            --hardware ${CMAKE_SOURCE_DIR}/notes/frontend/examples/hardware_2x2.json
            --simulation ${CMAKE_SOURCE_DIR}/llm/test/sram/simulation.json
            --mapping ${CMAKE_SOURCE_DIR}/llm/test/default/mapping.spec
            --runtime-root ${program_runtime_dir}
            --evidence-root ${program_runtime_dir}/swizzle-scale-runtime-evidence
            --report ${program_runtime_dir}/swizzle-scale-runtime-report.json
            --timeout 600)
    set_tests_properties(swizzle_scale_runtime_comparison PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        TIMEOUT 1800
        RUN_SERIAL TRUE
        LABELS "frontend;program;swizzle;scale;runtime;comparison;interop"
        PASS_REGULAR_EXPRESSION
            "\\[SWIZZLE SCALE RUNTIME\\] PASS: performance_benefit=0 measurement_complete=0 cases=3 branches=6 repeat=2"
        FAIL_REGULAR_EXPRESSION
            "\\[PROTO_WAIT\\]|Swizzle scale runtime first failure|Traceback")

    set(stage1a_state_runtime_runner
        "${CMAKE_SOURCE_DIR}/llm/test/frontend/integration/run_stage1a_state_cases.py")
    foreach(stage1a_case IN ITEMS p1 k1 pd1)
        string(TOUPPER "${stage1a_case}" stage1a_case_upper)
        add_test(NAME stage1a_${stage1a_case}_runtime
            COMMAND ${CMAKE_COMMAND} -E env
                PYTHONPATH=${CMAKE_SOURCE_DIR}
                ${PYTHON_EXECUTABLE} -B ${stage1a_state_runtime_runner}
                --case ${stage1a_case}
                --npusim $<TARGET_FILE:npusim>
                --finalizer $<TARGET_FILE:npusim_program_finalizer>
                --resolver $<TARGET_FILE:npusim_program_io_selftest>
                --hardware ${CMAKE_SOURCE_DIR}/llm/test/sram/hardware_numa.json
                --simulation ${CMAKE_SOURCE_DIR}/llm/test/sram/simulation.json
                --mapping ${CMAKE_SOURCE_DIR}/llm/test/default/mapping.spec
                --runtime-root ${program_runtime_dir}
                --report-root ${program_runtime_dir}/stage1a-reports)
        set_tests_properties(stage1a_${stage1a_case}_runtime PROPERTIES
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            TIMEOUT 300
            RUN_SERIAL TRUE
            LABELS "frontend;program;stage1a;state;runtime;${stage1a_case}"
            PASS_REGULAR_EXPRESSION "\\[STAGE1A ${stage1a_case_upper}\\] PASS"
            FAIL_REGULAR_EXPRESSION "\\[PROTO_WAIT\\]")
    endforeach()

    set(stage2_dense_forward_runner
        "${CMAKE_SOURCE_DIR}/llm/test/frontend/integration/run_stage2_dense_forward.py")
    foreach(stage2_tp IN ITEMS 1 2 4)
        if(stage2_tp EQUAL 1)
            set(stage2_timeout 300)
        elseif(stage2_tp EQUAL 2)
            set(stage2_timeout 600)
        else()
            set(stage2_timeout 1200)
        endif()
        add_test(NAME stage2_dense_forward_tp${stage2_tp}_runtime
            COMMAND ${CMAKE_COMMAND} -E env
                PYTHONPATH=${CMAKE_SOURCE_DIR}
                ${PYTHON_EXECUTABLE} -B ${stage2_dense_forward_runner}
                --tp-degree ${stage2_tp}
                --npusim $<TARGET_FILE:npusim>
                --finalizer $<TARGET_FILE:npusim_program_finalizer>
                --resolver $<TARGET_FILE:npusim_program_io_selftest>
                --simulation ${CMAKE_SOURCE_DIR}/llm/test/sram/simulation.json
                --runtime-root ${program_runtime_dir}
                --timeout ${stage2_timeout})
        set_tests_properties(stage2_dense_forward_tp${stage2_tp}_runtime PROPERTIES
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            TIMEOUT ${stage2_timeout}
            RUN_SERIAL TRUE
            LABELS "frontend;program;n6;program-io;stage2;runtime;tp${stage2_tp}"
            PASS_REGULAR_EXPRESSION "\\[STAGE2 TP${stage2_tp}\\] PASS"
            FAIL_REGULAR_EXPRESSION "\\[PROTO_WAIT\\]")
    endforeach()

    set(stage4_pd_pre_runtime_runner
        "${CMAKE_SOURCE_DIR}/llm/test/frontend/integration/run_stage4_pd_pre_runtime.py")
    add_test(NAME stage4_pds_tp1_pre_runtime
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_SOURCE_DIR}
            ${PYTHON_EXECUTABLE} -B ${stage4_pd_pre_runtime_runner}
            --finalizer $<TARGET_FILE:npusim_program_finalizer>
            --resolver $<TARGET_FILE:npusim_program_io_selftest>
            --runtime-root ${program_runtime_dir})
    set_tests_properties(stage4_pds_tp1_pre_runtime PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        TIMEOUT 180
        RUN_SERIAL TRUE
        LABELS "frontend;program;n6;program-io;stage4;pds;pre-runtime"
        PASS_REGULAR_EXPRESSION "\\[STAGE4 PDS TP1 PRE-RUNTIME\\] PASS"
        FAIL_REGULAR_EXPRESSION "\\[PROTO_WAIT\\]")

    set(stage4_pd_runtime_runner
        "${CMAKE_SOURCE_DIR}/llm/test/frontend/integration/run_stage4_pd_runtime.py")
    foreach(stage4_case IN ITEMS fused pds)
        string(TOUPPER "${stage4_case}" stage4_case_upper)
        add_test(NAME stage4_${stage4_case}_tp1_runtime
            COMMAND ${CMAKE_COMMAND} -E env
                PYTHONPATH=${CMAKE_SOURCE_DIR}
                ${PYTHON_EXECUTABLE} -B ${stage4_pd_runtime_runner}
                --case ${stage4_case}
                --npusim $<TARGET_FILE:npusim>
                --finalizer $<TARGET_FILE:npusim_program_finalizer>
                --resolver $<TARGET_FILE:npusim_program_io_selftest>
                --simulation ${CMAKE_SOURCE_DIR}/llm/test/sram/simulation.json
                --runtime-root ${program_runtime_dir}
                --report ${program_runtime_dir}/stage4-${stage4_case}-tp1-report.json
                --timeout 300)
        set_tests_properties(stage4_${stage4_case}_tp1_runtime PROPERTIES
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            TIMEOUT 300
            RUN_SERIAL TRUE
            LABELS "frontend;program;n6;program-io;stage4;${stage4_case};runtime"
            PASS_REGULAR_EXPRESSION "\\[STAGE4 ${stage4_case_upper} TP1\\] PASS"
            FAIL_REGULAR_EXPRESSION "\\[PROTO_WAIT\\]")
    endforeach()

    add_test(NAME stage4_pdr_tp2_to_tp1_runtime
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_SOURCE_DIR}
            ${PYTHON_EXECUTABLE} -B ${stage4_pd_runtime_runner}
            --case pdr
            --npusim $<TARGET_FILE:npusim>
            --finalizer $<TARGET_FILE:npusim_program_finalizer>
            --resolver $<TARGET_FILE:npusim_program_io_selftest>
            --simulation ${CMAKE_SOURCE_DIR}/llm/test/sram/simulation.json
            --runtime-root ${program_runtime_dir}
            --report ${program_runtime_dir}/stage4-pdr-tp2-to-tp1-report.json
            --timeout 600)
    set_tests_properties(stage4_pdr_tp2_to_tp1_runtime PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        TIMEOUT 600
        RUN_SERIAL TRUE
        LABELS "frontend;program;n6;program-io;stage4;pdr;runtime"
        PASS_REGULAR_EXPRESSION "\\[STAGE4 PDR TP2_TO_TP1\\] PASS"
        FAIL_REGULAR_EXPRESSION "\\[PROTO_WAIT\\]")

    set(train_forward_runtime_runner
        "${CMAKE_SOURCE_DIR}/llm/test/frontend/integration/run_train_forward_runtime.py")
    add_test(NAME train_forward_dp2_tp2_runtime
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_SOURCE_DIR}
            ${PYTHON_EXECUTABLE} -B ${train_forward_runtime_runner}
            --npusim $<TARGET_FILE:npusim>
            --finalizer $<TARGET_FILE:npusim_program_finalizer>
            --resolver $<TARGET_FILE:npusim_program_io_selftest>
            --simulation ${CMAKE_SOURCE_DIR}/llm/test/sram/simulation.json
            --runtime-root ${program_runtime_dir}
            --report ${program_runtime_dir}/train-forward-dp2-tp2-report.json
            --timeout 600)
    set_tests_properties(train_forward_dp2_tp2_runtime PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        TIMEOUT 600
        RUN_SERIAL TRUE
        LABELS "frontend;program;n6;program-io;train;runtime;dp2;tp2"
        PASS_REGULAR_EXPRESSION "\\[TRAIN FORWARD DP2 TP2\\] PASS"
        FAIL_REGULAR_EXPRESSION "\\[PROTO_WAIT\\]")

    set(s2_lite_dp2_rooted_ar_runtime_runner
        "${CMAKE_SOURCE_DIR}/llm/test/frontend/integration/run_lite_train_rooted_ar_runtime.py")
    add_test(NAME s2_lite_dp2_rooted_ar_runtime
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_SOURCE_DIR}
            ${PYTHON_EXECUTABLE} -B ${s2_lite_dp2_rooted_ar_runtime_runner}
            --npusim $<TARGET_FILE:npusim>
            --finalizer $<TARGET_FILE:npusim_program_finalizer>
            --resolver $<TARGET_FILE:npusim_program_io_selftest>
            --simulation ${CMAKE_SOURCE_DIR}/llm/test/sram/simulation.json
            --runtime-root ${program_runtime_dir}
            --timeout 600)
    set_tests_properties(s2_lite_dp2_rooted_ar_runtime PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        TIMEOUT 600
        RUN_SERIAL TRUE
        LABELS "frontend;program;n6;program-io;s2-lite;train;dp2;tp1;rooted-ar;runtime"
        PASS_REGULAR_EXPRESSION "\\[S2-LITE DP2 ROOTED-AR\\] PASS"
        FAIL_REGULAR_EXPRESSION "\\[PROTO_WAIT\\]")

    set(s2_lite_dp4_tree_ar_runtime_runner
        "${CMAKE_SOURCE_DIR}/llm/test/frontend/integration/run_lite_train_dp4_tree_ar_runtime.py")
    add_test(NAME s2_lite_dp4_tree_ar_runtime
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_SOURCE_DIR}
            ${PYTHON_EXECUTABLE} -B ${s2_lite_dp4_tree_ar_runtime_runner}
            --npusim $<TARGET_FILE:npusim>
            --finalizer $<TARGET_FILE:npusim_program_finalizer>
            --resolver $<TARGET_FILE:npusim_program_io_selftest>
            --simulation ${CMAKE_SOURCE_DIR}/llm/test/sram/simulation.json
            --runtime-root ${program_runtime_dir}
            --timeout 900)
    set_tests_properties(s2_lite_dp4_tree_ar_runtime PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        TIMEOUT 900
        RUN_SERIAL TRUE
        LABELS "frontend;program;n6;program-io;s2-lite;train;backward;dp4;tp1;tree-ar;runtime"
        PASS_REGULAR_EXPRESSION "\\[S2-LITE DP4 TREE-AR\\] PASS"
        FAIL_REGULAR_EXPRESSION "\\[PROTO_WAIT\\]")

    set(s3_lite_moe_backward_runtime_runner
        "${CMAKE_SOURCE_DIR}/llm/test/frontend/integration/run_lite_moe_backward_runtime.py")
    add_test(NAME s3_lite_moe_backward_runtime
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_SOURCE_DIR}:${CMAKE_SOURCE_DIR}/llm/test/frontend/integration
            ${PYTHON_EXECUTABLE} -B ${s3_lite_moe_backward_runtime_runner}
            --npusim $<TARGET_FILE:npusim>
            --finalizer $<TARGET_FILE:npusim_program_finalizer>
            --resolver $<TARGET_FILE:npusim_program_io_selftest>
            --simulation ${CMAKE_SOURCE_DIR}/llm/test/sram/simulation.json
            --runtime-root ${program_runtime_dir}
            --timeout 600)
    set_tests_properties(s3_lite_moe_backward_runtime PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        TIMEOUT 600
        RUN_SERIAL TRUE
        LABELS "frontend;program;n6;program-io;s3-lite;moe;backward;ep2;runtime"
        PASS_REGULAR_EXPRESSION "\\[S3-LITE MOE BACKWARD\\] PASS"
        FAIL_REGULAR_EXPRESSION "\\[PROTO_WAIT\\]")

    set(s3_lite_dp4_runtime_matrix_runner
        "${CMAKE_SOURCE_DIR}/llm/test/frontend/integration/run_lite_moe_dp4_runtime_matrix.py")
    add_test(NAME s3_lite_dp4_runtime_matrix
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_SOURCE_DIR}:${CMAKE_SOURCE_DIR}/llm/test/frontend/integration
            ${PYTHON_EXECUTABLE} -B ${s3_lite_dp4_runtime_matrix_runner}
            --npusim $<TARGET_FILE:npusim>
            --finalizer $<TARGET_FILE:npusim_program_finalizer>
            --resolver $<TARGET_FILE:npusim_program_io_selftest>
            --simulation ${CMAKE_SOURCE_DIR}/llm/test/sram/simulation.json
            --runtime-root ${program_runtime_dir}
            --timeout 900)
    set_tests_properties(s3_lite_dp4_runtime_matrix PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        TIMEOUT 900
        RUN_SERIAL TRUE
        LABELS "frontend;program;n6;program-io;s3-lite;moe;dp4;ep4;matrix;runtime"
        PASS_REGULAR_EXPRESSION "\\[S3-LITE DP4 MATRIX\\] PASS"
        FAIL_REGULAR_EXPRESSION "\\[PROTO_WAIT\\]|\\[S3-LITE MOE DP4\\] FAIL")

    set(stage3_static_profiles_runner
        "${CMAKE_SOURCE_DIR}/llm/test/frontend/integration/run_stage3_static_profiles.py")
    foreach(stage3_case IN ITEMS prefill mixed decode)
        string(TOUPPER "${stage3_case}" stage3_case_upper)
        add_test(NAME stage3_static_${stage3_case}_runtime
            COMMAND ${CMAKE_COMMAND} -E env
                PYTHONPATH=${CMAKE_SOURCE_DIR}
                ${PYTHON_EXECUTABLE} -B ${stage3_static_profiles_runner}
                --case ${stage3_case}
                --npusim $<TARGET_FILE:npusim>
                --finalizer $<TARGET_FILE:npusim_program_finalizer>
                --resolver $<TARGET_FILE:npusim_program_io_selftest>
                --simulation ${CMAKE_SOURCE_DIR}/llm/test/sram/simulation.json
                --runtime-root ${program_runtime_dir}
                --timeout 300)
        set_tests_properties(stage3_static_${stage3_case}_runtime PROPERTIES
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            TIMEOUT 300
            RUN_SERIAL TRUE
            LABELS "frontend;program;n6;program-io;stage3;runtime;${stage3_case}"
            PASS_REGULAR_EXPRESSION "\\[STAGE3 ${stage3_case_upper}\\] PASS"
            FAIL_REGULAR_EXPRESSION "\\[PROTO_WAIT\\]")
    endforeach()

    set(stage3_static_profile_negative_runner
        "${CMAKE_SOURCE_DIR}/llm/test/frontend/integration/run_stage3_static_profile_negative_evidence.py")
    add_test(NAME stage3_static_profile_negative_evidence
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_SOURCE_DIR}
            ${PYTHON_EXECUTABLE} -B ${stage3_static_profile_negative_runner})
    set_tests_properties(stage3_static_profile_negative_evidence PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        TIMEOUT 60
        RUN_SERIAL TRUE
        LABELS "frontend;stage3;negative"
        PASS_REGULAR_EXPRESSION "\\[STAGE3 NEGATIVE\\] PASS: witnesses=9 deterministic=1")

    set(stage2_dense_forward_negative_runner
        "${CMAKE_SOURCE_DIR}/llm/test/frontend/integration/run_stage2_dense_forward_negative_evidence.py")
    add_test(NAME stage2_dense_forward_negative_evidence
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_SOURCE_DIR}
            ${PYTHON_EXECUTABLE} -B ${stage2_dense_forward_negative_runner}
            --finalizer $<TARGET_FILE:npusim_program_finalizer>
            --resolver $<TARGET_FILE:npusim_program_io_selftest>
            --runtime-root ${program_runtime_dir})
    set_tests_properties(stage2_dense_forward_negative_evidence PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        TIMEOUT 180
        RUN_SERIAL TRUE
        LABELS "frontend;program;program-io;stage2;negative"
        PASS_REGULAR_EXPRESSION "\\[STAGE2 NEGATIVE\\] PASS: witnesses=9 deterministic=1")


    set(naive_n7_e1_runner
        "${CMAKE_SOURCE_DIR}/llm/test/frontend/integration/run_naive_e1.py")
    add_test(NAME naive_n7_e1_runtime
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_SOURCE_DIR}
            ${PYTHON_EXECUTABLE} -B ${naive_n7_e1_runner}
            --npusim $<TARGET_FILE:npusim>
            --finalizer $<TARGET_FILE:npusim_program_finalizer>
            --spec ${CMAKE_SOURCE_DIR}/notes/frontend/examples/naive_dense_tp2.yaml
            --hardware ${CMAKE_SOURCE_DIR}/notes/frontend/examples/hardware_2x1.json
            --simulation ${CMAKE_SOURCE_DIR}/llm/test/sram/simulation.json
            --mapping ${CMAKE_SOURCE_DIR}/llm/test/default/mapping.spec
            --runtime-root ${program_runtime_dir})
    set_tests_properties(naive_n7_e1_runtime PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        TIMEOUT 300
        RUN_SERIAL TRUE
        LABELS "frontend;program;n7;e1;runtime;production"
        PASS_REGULAR_EXPRESSION "\\[NAIVE N7 E1\\] PASS"
        FAIL_REGULAR_EXPRESSION "\\[PROTO_WAIT\\]")

    if(BUILD_N7_NIGHTLY_TESTS)
        set(naive_n7_e2_runner
            "${CMAKE_SOURCE_DIR}/llm/test/frontend/integration/run_naive_e2.py")
        add_test(NAME naive_n7_e2_runtime
            COMMAND ${CMAKE_COMMAND} -E env
                PYTHONPATH=${CMAKE_SOURCE_DIR}
                ${PYTHON_EXECUTABLE} -B ${naive_n7_e2_runner}
                --npusim $<TARGET_FILE:npusim>
                --finalizer $<TARGET_FILE:npusim_program_finalizer>
                --spec ${CMAKE_SOURCE_DIR}/notes/frontend/examples/naive_dense_tp4.yaml
                --hardware ${CMAKE_SOURCE_DIR}/notes/frontend/examples/hardware_2x2.json
                --simulation ${CMAKE_SOURCE_DIR}/llm/test/sram/simulation.json
                --mapping ${CMAKE_SOURCE_DIR}/llm/test/default/mapping.spec
                --runtime-root ${program_runtime_dir})
        set_tests_properties(naive_n7_e2_runtime PROPERTIES
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            TIMEOUT 900
            RUN_SERIAL TRUE
            LABELS "frontend;program;n7;e2;runtime;nightly"
            PASS_REGULAR_EXPRESSION "\\[NAIVE N7 E2\\] PASS"
            FAIL_REGULAR_EXPRESSION "\\[PROTO_WAIT\\]")
    endif()

    add_test(NAME frontend_python_unit
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_SOURCE_DIR}
            ${PYTHON_EXECUTABLE} -B -m unittest discover
            -s ${CMAKE_SOURCE_DIR}/llm/test/frontend/unit
            -p test_*.py
            -v)
    set_tests_properties(frontend_python_unit PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        TIMEOUT 180
        LABELS "frontend;n1;unit;contracts")

    set(frontend_n0_backend_runner
        "${CMAKE_SOURCE_DIR}/llm/test/frontend/integration/run_n0_backend_gate.py")
    add_test(NAME frontend_n0_backend_gate
        COMMAND ${PYTHON_EXECUTABLE} ${frontend_n0_backend_runner}
            --npusim $<TARGET_FILE:npusim>
            --program-fixture $<TARGET_FILE:npusim_program_fixture>
            --hardware ${CMAKE_SOURCE_DIR}/llm/test/sram/hardware_numa.json
            --simulation ${CMAKE_SOURCE_DIR}/llm/test/sram/simulation.json
            --mapping ${CMAKE_SOURCE_DIR}/llm/test/default/mapping.spec
            --runtime-root ${CMAKE_BINARY_DIR})
    set_tests_properties(frontend_n0_backend_gate PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        TIMEOUT 300
        RUN_SERIAL TRUE
        LABELS "frontend;n0;backend;program;p2p;runtime;numerical"
        PASS_REGULAR_EXPRESSION
            "\\[FRONTEND N0 BACKEND\\] PASS: 2-die TP2 dual-owner RS"
        FAIL_REGULAR_EXPRESSION "\\[PROTO_WAIT\\]")

    function(add_p5_endpoint_fixture fixture_name fixture_option)
        set(fixture_path
            "${program_runtime_dir}/p5_${fixture_name}.npup")
        add_test(NAME program_p5_${fixture_name}_fixture_generate
            COMMAND npusim_program_fixture
                "${fixture_path}" "${fixture_option}")
        set_tests_properties(
            program_p5_${fixture_name}_fixture_generate PROPERTIES
            WORKING_DIRECTORY "${program_runtime_dir}"
            LABELS "isa;program;p2p;fixture")
    endfunction()

    add_p5_endpoint_fixture(sram_sync_129 --p5-sram-sync-129)
    add_p5_endpoint_fixture(sram_async_4k --p5-sram-async-4k)
    add_p5_endpoint_fixture(hbm_sync_129 --p5-hbm-sync-129)
    add_p5_endpoint_fixture(pair_missing --p5-pair-missing)
    add_p5_endpoint_fixture(pair_mismatch --p5-pair-mismatch)

    set(p5_endpoint_runner
        "${CMAKE_SOURCE_DIR}/llm/test/program/run_p5_endpoint_program.py")
    set(p5_endpoint_oracle
        "${CMAKE_SOURCE_DIR}/llm/test/program/p5_endpoint_oracle.json")
    set(p5_endpoint_hardware
        "${CMAKE_SOURCE_DIR}/llm/test/sram/hardware_numa.json")

    add_test(NAME program_p5_endpoint_runner_selftest
        COMMAND ${PYTHON_EXECUTABLE} ${p5_endpoint_runner} --selftest)
    set_tests_properties(program_p5_endpoint_runner_selftest PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        LABELS "isa;program;p2p;fixture"
        PASS_REGULAR_EXPRESSION
            "\\[P5 ENDPOINT PROGRAM\\] PASS: manifest/category selftest")

    add_test(NAME program_p5_endpoint_loader_negatives
        COMMAND ${PYTHON_EXECUTABLE} ${p5_endpoint_runner}
            --npusim $<TARGET_FILE:npusim>
            --program-fixture $<TARGET_FILE:npusim_program_fixture>
            --oracle ${p5_endpoint_oracle}
            --hardware ${p5_endpoint_hardware}
            --simulation ${CMAKE_SOURCE_DIR}/llm/test/sram/simulation.json
            --mapping ${CMAKE_SOURCE_DIR}/llm/test/default/mapping.spec
            --runtime-root ${CMAKE_BINARY_DIR}
            --negative-only)
    set_tests_properties(program_p5_endpoint_loader_negatives PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        TIMEOUT 120
        RUN_SERIAL TRUE
        LABELS "isa;program;p2p;negative;loader"
        PASS_REGULAR_EXPRESSION
            "\\[P5 ENDPOINT PROGRAM\\] PASS: loader negatives"
        FAIL_REGULAR_EXPRESSION "Loaded Program Format")

    set(p5_endpoint_simulation
        "${CMAKE_SOURCE_DIR}/llm/test/sram/simulation.json")
    set(p5_endpoint_streaming_simulation
        "${CMAKE_SOURCE_DIR}/llm/test/program/p5_streaming_simulation.json")
    set(p5_endpoint_bounded_hardware
        "${CMAKE_SOURCE_DIR}/llm/test/program/p5_bounded_hardware.json")
    set(p5_endpoint_large_hardware
        "${CMAKE_SOURCE_DIR}/llm/test/program/p5_large_hardware.json")
    set(p5_endpoint_behavioral_hardware
        "${CMAKE_SOURCE_DIR}/llm/test/program/p5_behavioral_hardware.json")
    set(p5_endpoint_behavioral_simulation
        "${CMAKE_SOURCE_DIR}/llm/test/program/p5_behavioral_simulation.json")

    # The production endpoint matrix is serialized because the runner shares
    # trace/VCD output paths across simulations.
    function(add_p5_endpoint_runtime variant)
        add_test(NAME program_p5_endpoint_runtime_${variant}
            COMMAND ${PYTHON_EXECUTABLE} ${p5_endpoint_runner}
                --npusim $<TARGET_FILE:npusim>
                --program-fixture $<TARGET_FILE:npusim_program_fixture>
                --oracle ${p5_endpoint_oracle}
                --hardware ${p5_endpoint_hardware}
                --simulation ${p5_endpoint_simulation}
                --mapping ${CMAKE_SOURCE_DIR}/llm/test/default/mapping.spec
                --runtime-root ${CMAKE_BINARY_DIR}
                --large-hardware ${p5_endpoint_large_hardware}
                --variant ${variant}
                ${ARGN})
        set_tests_properties(
            program_p5_endpoint_runtime_${variant} PROPERTIES
            WORKING_DIRECTORY "${program_runtime_dir}"
            TIMEOUT 1200
            RUN_SERIAL TRUE
            LABELS
                "isa;program;p2p;runtime;${variant}"
            PASS_REGULAR_EXPRESSION
                "\\[P5 ENDPOINT PROGRAM\\] PASS: runtime positives"
            FAIL_REGULAR_EXPRESSION "\\[PROTO_WAIT\\]")
    endfunction()

    add_p5_endpoint_runtime(base)
    add_p5_endpoint_runtime(streaming
        --streaming-simulation ${p5_endpoint_streaming_simulation})
    add_p5_endpoint_runtime(bounded
        --bounded-hardware ${p5_endpoint_bounded_hardware})
    add_p5_endpoint_runtime(behavioral
        --behavioral-hardware ${p5_endpoint_behavioral_hardware}
        --behavioral-simulation ${p5_endpoint_behavioral_simulation})

    set(p6_collective_runner
        "${CMAKE_SOURCE_DIR}/llm/test/program/run_p6_collective_program.py")
    set(p6_collective_oracle
        "${CMAKE_SOURCE_DIR}/llm/test/program/p6_collective_oracle.json")
    set(p6_collective_hardware
        "${CMAKE_SOURCE_DIR}/llm/test/program/p6_collective_hardware.json")

    add_test(NAME program_p6_collective_runner_selftest
        COMMAND ${PYTHON_EXECUTABLE} ${p6_collective_runner} --selftest)
    set_tests_properties(program_p6_collective_runner_selftest PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        LABELS "isa;program;collective;p6;fixture"
        PASS_REGULAR_EXPRESSION
            "\\[P6 COLLECTIVE PROGRAM\\] PASS: trace/stats/drain/oracle selftest")

    add_test(NAME program_p6_collective_fixture_matrix
        COMMAND ${PYTHON_EXECUTABLE} ${p6_collective_runner}
            --fixture-selftest
            --program-fixture $<TARGET_FILE:npusim_program_fixture>
            --npusim $<TARGET_FILE:npusim>
            --oracle ${p6_collective_oracle}
            --hardware ${p6_collective_hardware}
            --simulation ${CMAKE_SOURCE_DIR}/llm/test/sram/simulation.json
            --mapping ${CMAKE_SOURCE_DIR}/llm/test/default/mapping.spec
            --runtime-root ${CMAKE_BINARY_DIR})
    set_tests_properties(program_p6_collective_fixture_matrix PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        TIMEOUT 300
        RUN_SERIAL TRUE
        LABELS "isa;program;collective;p6;fixture"
        PASS_REGULAR_EXPRESSION
            "\\[P6 COLLECTIVE PROGRAM\\] PASS: 37 positive \\+ 6 negative artifacts \\+ 1 fixture rejection \\+ 6 loader rejections")

    add_test(NAME program_p6_collective_runtime_matrix
        COMMAND ${PYTHON_EXECUTABLE} ${p6_collective_runner}
            --runtime-all
            --npusim $<TARGET_FILE:npusim>
            --program-fixture $<TARGET_FILE:npusim_program_fixture>
            --oracle ${p6_collective_oracle}
            --hardware ${p6_collective_hardware}
            --simulation ${CMAKE_SOURCE_DIR}/llm/test/sram/simulation.json
            --mapping ${CMAKE_SOURCE_DIR}/llm/test/default/mapping.spec
            --runtime-root ${CMAKE_BINARY_DIR})
    set_tests_properties(program_p6_collective_runtime_matrix PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        TIMEOUT 3600
        RUN_SERIAL TRUE
        LABELS "isa;program;collective;p6;runtime"
        PASS_REGULAR_EXPRESSION
            "\\[P6 COLLECTIVE PROGRAM\\] PASS: runtime matrix 37 fixtures"
        FAIL_REGULAR_EXPRESSION "\\[PROTO_WAIT\\]")

    set(p7_accelerated_32k_runner
        "${CMAKE_SOURCE_DIR}/llm/test/program/run_p7_accelerated_32k_program.py")
    set(p7_accelerated_32k_oracle
        "${CMAKE_SOURCE_DIR}/llm/test/program/p7_accelerated_32k_oracle.json")

    add_test(NAME program_p7_accelerated_32k_runtime_matrix
        COMMAND ${PYTHON_EXECUTABLE} ${p7_accelerated_32k_runner}
            --npusim $<TARGET_FILE:npusim>
            --program-fixture $<TARGET_FILE:npusim_program_fixture>
            --oracle ${p7_accelerated_32k_oracle}
            --p6-oracle ${p6_collective_oracle}
            --hardware ${p6_collective_hardware}
            --simulation ${CMAKE_SOURCE_DIR}/llm/test/sram/simulation.json
            --mapping ${CMAKE_SOURCE_DIR}/llm/test/default/mapping.spec
            --runtime-root ${CMAKE_BINARY_DIR})
    set_tests_properties(program_p7_accelerated_32k_runtime_matrix PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        TIMEOUT 3600
        RUN_SERIAL TRUE
        LABELS "isa;program;collective;p7;runtime;profile;32k"
        PASS_REGULAR_EXPRESSION
            "\\[P7 ACCELERATED 32K\\] PASS: cases=2 profiles=4 isolated_processes=8 K=1"
        FAIL_REGULAR_EXPRESSION
            "\\[PROTO_WAIT\\]|requested collective backend rejected|fallback")

    set(p8_double_buffer_runner
        "${CMAKE_SOURCE_DIR}/llm/test/isa/run_p8_double_buffer_program.py")
    set(p8_double_buffer_oracle
        "${CMAKE_SOURCE_DIR}/llm/test/isa/p8_double_buffer_oracle.json")
    set(p8_double_buffer_hardware
        "${CMAKE_SOURCE_DIR}/llm/test/isa/p8_double_buffer_hardware.json")

    add_test(NAME program_p8_double_buffer_runner_selftest
        COMMAND ${PYTHON_EXECUTABLE} ${p8_double_buffer_runner}
            --selftest --oracle ${p8_double_buffer_oracle})
    set_tests_properties(program_p8_double_buffer_runner_selftest PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        LABELS "isa;program;p8;fixture;memory"
        PASS_REGULAR_EXPRESSION
            "\\[P8 DOUBLE BUFFER\\] PASS: runner/oracle selftest")

    add_test(NAME program_p8_double_buffer_runtime
        COMMAND ${PYTHON_EXECUTABLE} ${p8_double_buffer_runner}
            --npusim $<TARGET_FILE:npusim>
            --program-fixture $<TARGET_FILE:npusim_program_fixture>
            --oracle ${p8_double_buffer_oracle}
            --hardware ${p8_double_buffer_hardware}
            --simulation ${CMAKE_SOURCE_DIR}/llm/test/sram/simulation.json
            --mapping ${CMAKE_SOURCE_DIR}/llm/test/default/mapping.spec
            --runtime-root ${CMAKE_BINARY_DIR})
    set_tests_properties(program_p8_double_buffer_runtime PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        TIMEOUT 180
        RUN_SERIAL TRUE
        LABELS "isa;program;p8;runtime;memory;double-buffer"
        PASS_REGULAR_EXPRESSION
            "\\[P8 DOUBLE BUFFER\\] PASS: exact token lifecycle and full drain"
        FAIL_REGULAR_EXPRESSION "\\[PROTO_WAIT\\]")

    if(TARGET npusim_runtime_sanitizer)
        add_test(NAME program_p8_double_buffer_runtime_sanitizer
            COMMAND ${PYTHON_EXECUTABLE} ${p8_double_buffer_runner}
                --npusim $<TARGET_FILE:npusim_runtime_sanitizer>
                --program-fixture $<TARGET_FILE:npusim_program_fixture>
                --oracle ${p8_double_buffer_oracle}
                --hardware ${p8_double_buffer_hardware}
                --simulation ${CMAKE_SOURCE_DIR}/llm/test/sram/simulation.json
                --mapping ${CMAKE_SOURCE_DIR}/llm/test/default/mapping.spec
                --runtime-root ${CMAKE_BINARY_DIR})
        set_tests_properties(program_p8_double_buffer_runtime_sanitizer
            PROPERTIES
            WORKING_DIRECTORY "${program_runtime_dir}"
            TIMEOUT 300
            RUN_SERIAL TRUE
            LABELS
                "isa;program;p8;runtime;memory;double-buffer;sanitizer"
            PASS_REGULAR_EXPRESSION
                "\\[P8 DOUBLE BUFFER\\] PASS: exact token lifecycle and full drain"
            FAIL_REGULAR_EXPRESSION
                "\\[PROTO_WAIT\\]|AddressSanitizer|runtime error:|LeakSanitizer"
            ENVIRONMENT
                "ASAN_OPTIONS=detect_leaks=1:strict_string_checks=1:halt_on_error=1:abort_on_error=1;LSAN_OPTIONS=exitcode=23:report_objects=1;UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1")
    endif()

    set(p8_program_b_runner
        "${CMAKE_SOURCE_DIR}/llm/test/isa/run_p8_program_b.py")
    set(p8_program_b_oracle
        "${CMAKE_SOURCE_DIR}/llm/test/isa/p8_program_b_oracle.json")
    set(p8_program_b_hardware
        "${CMAKE_SOURCE_DIR}/llm/test/isa/p8_program_b_hardware.json")

    add_test(NAME program_p8_program_b_runner_selftest
        COMMAND ${PYTHON_EXECUTABLE} ${p8_program_b_runner}
            --selftest --oracle ${p8_program_b_oracle})
    set_tests_properties(program_p8_program_b_runner_selftest PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        LABELS "isa;program;p8;fixture;collective"
        PASS_REGULAR_EXPRESSION
            "\\[P8 PROGRAM B\\] PASS: artifact/byte/profile/repeat runner selftest")

    add_test(NAME program_p8_program_b_runtime
        COMMAND ${PYTHON_EXECUTABLE} ${p8_program_b_runner}
            --npusim $<TARGET_FILE:npusim>
            --program-fixture $<TARGET_FILE:npusim_program_fixture>
            --oracle ${p8_program_b_oracle}
            --hardware ${p8_program_b_hardware}
            --simulation ${CMAKE_SOURCE_DIR}/llm/test/sram/simulation.json
            --mapping ${CMAKE_SOURCE_DIR}/llm/test/default/mapping.spec
            --runtime-root ${CMAKE_BINARY_DIR}
            --profiles baseline,broadcast_only,reduce_only,reduce_broadcast
            --repeat 3
            --require-accelerated)
    set_tests_properties(program_p8_program_b_runtime PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        TIMEOUT 1800
        RUN_SERIAL TRUE
        LABELS "isa;program;p8;runtime;collective;profile"
        PASS_REGULAR_EXPRESSION
            "\\[P8 PROGRAM B\\] PASS: profiles=baseline,broadcast_only,reduce_only,reduce_broadcast repeat=3 isolated_processes=12"
        FAIL_REGULAR_EXPRESSION "\\[PROTO_WAIT\\]")

    add_test(NAME program_isa_v1_mutation_gate
        COMMAND ${PYTHON_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/llm/test/isa/run_isa_program_mutation.py
            --npusim $<TARGET_FILE:npusim>
            --program-fixture $<TARGET_FILE:npusim_program_fixture>
            --runtime-root ${CMAKE_BINARY_DIR})
    set_tests_properties(program_isa_v1_mutation_gate PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        TIMEOUT 180
        RUN_SERIAL TRUE
        LABELS "isa;program;p8;negative;mutation"
        PASS_REGULAR_EXPRESSION
            "\\[ISA MUTATION\\] PASS: 58 deterministic container/record mutations")

    add_test(NAME program_legacy_compute_diff
        COMMAND ${PYTHON_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/llm/test/program/run_compute_chain_diff.py
            --npusim $<TARGET_FILE:npusim>
            --program-fixture $<TARGET_FILE:npusim_program_fixture>
            --legacy-workload
                ${CMAKE_SOURCE_DIR}/llm/test/program/compute_chain_legacy.json
            --hardware
                ${CMAKE_SOURCE_DIR}/llm/test/program/compute_chain_hardware.json
            --simulation ${CMAKE_SOURCE_DIR}/llm/test/default/simulation.json
            --mapping ${CMAKE_SOURCE_DIR}/llm/test/default/mapping.spec
            --runtime-root ${CMAKE_BINARY_DIR})
    set_tests_properties(program_legacy_compute_diff PROPERTIES
        WORKING_DIRECTORY "${program_runtime_dir}"
        TIMEOUT 120
        RUN_SERIAL TRUE
        LABELS "isa;program;runtime;sram-bind;compute-cost;legacy"
        PASS_REGULAR_EXPRESSION
            "\\[P3 COMPUTE DIFF\\] PASS: lifecycle matches with no temporary residual"
        FAIL_REGULAR_EXPRESSION "\\[PROTO_WAIT\\]")

    foreach(hbm_stage RANGE 0 4)
        add_test(NAME hbm_r${hbm_stage}_selftest
                 COMMAND npusim --hbm-r${hbm_stage}-selftest)
        set_tests_properties(hbm_r${hbm_stage}_selftest PROPERTIES
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            LABELS "hbm;distributed-memory")
    endforeach()
    foreach(sram_stage RANGE 0 6)
        add_test(NAME sram_r${sram_stage}_selftest
                 COMMAND npusim --sram-r${sram_stage}-selftest)
        set_tests_properties(sram_r${sram_stage}_selftest PROPERTIES
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            LABELS "sram;on-chip-memory")
    endforeach()

    add_test(NAME workload_rendezvous_selftest
             COMMAND npusim --workload-rendezvous-selftest)
    set_tests_properties(workload_rendezvous_selftest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        LABELS "config;startup")

    add_test(NAME default_workload_smoke COMMAND npusim --trace-window 1000000)
    set_tests_properties(default_workload_smoke PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        TIMEOUT 60
        LABELS "config;startup;legacy"
        PASS_REGULAR_EXPRESSION "\\[START_DATA\\] enqueued=1 injected=1 accepted=1 delivered=1 consumed=1 completed=1"
        FAIL_REGULAR_EXPRESSION "\\[PROTO_WAIT\\]")

    add_test(NAME start_data_four_source_long
             COMMAND npusim
                --workload-config ${CMAKE_SOURCE_DIR}/llm/test/start_data/workload_4source_long.json
                --hardware-config ${CMAKE_SOURCE_DIR}/llm/test/start_data/hardware_4x4.json
                --simulation-config ${CMAKE_SOURCE_DIR}/llm/test/default/simulation.json
                --mapping-config ${CMAKE_SOURCE_DIR}/llm/test/default/mapping.spec
                --trace-window 1000000)
    set_tests_properties(start_data_four_source_long PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        TIMEOUT 60
        LABELS "startup;host-lane;legacy"
        PASS_REGULAR_EXPRESSION "\\[START_DATA\\] enqueued=4 injected=4 accepted=4 delivered=4 consumed=4 completed=4"
        FAIL_REGULAR_EXPRESSION "\\[PROTO_WAIT\\]")

    add_test(NAME playground_legacy_baseline
             COMMAND npusim
                --workload-config ${CMAKE_SOURCE_DIR}/llm/test/workload_config/playground.json
                --hardware-config ${CMAKE_SOURCE_DIR}/llm/test/hardware_config/default/8x8.json
                --simulation-config ${CMAKE_SOURCE_DIR}/llm/test/simulation_config/default_spec.json
                --mapping-config ${CMAKE_SOURCE_DIR}/llm/test/mapping_config/default_mapping.spec
                --trace-window 1000000)
    set_tests_properties(playground_legacy_baseline PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        TIMEOUT 120
        LABELS "startup;legacy;full-workload"
        PASS_REGULAR_EXPRESSION "\\[START_DATA\\] enqueued=4 injected=4 accepted=4 delivered=4 consumed=4 completed=4"
        FAIL_REGULAR_EXPRESSION "\\[PROTO_WAIT\\]")
endif()
# add_test_executable(load_config  "./llm/unittest/load_config.cpp")      
# add_test_executable(global_chip_test  "./llm/unittest/global_chip_test.cpp")

# function(add_debug_executable target_name test_main_file)
#     add_executable(${target_name}_debug ${SRC_FILES} ${test_main_file})
#     target_link_libraries(${target_name}_debug
#         PUBLIC
#             systemc 
#             cairo 
#             sfml-graphics 
#             sfml-system 
#             sfml-window 
#             m 
#             pthread
#             DRAMSys::libdramsys
#             DRAMSys::util
#             DRAMSys::config
#     )
#     target_compile_options(${target_name}_debug PRIVATE 
#         -g -O0 -Wall -Wextra -DBROAD_W=16 -DDEBUG
#     )
# endfunction()

# 添加Debug构建目标
# add_debug_executable(npusim       "./llm/unittest/npusim.cpp")
# add_debug_executable(load_config  "./llm/unittest/load_config.cpp")
# add_debug_executable(global_chip_test  "./llm/unittest/global_chip_test.cpp")

# 更新清理目标
# set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "npusim;npusim_debug;load_config;load_config_debug")


# add_executable(npusim ${SRC_FILES})
# target_link_libraries(npusim
#     PUBLIC
#     systemc cairo sfml-graphics sfml-system sfml-window m pthread
#         DRAMSys::libdramsys
#         DRAMSys::util
#         DRAMSys::config
# )
# target_compile_options(npusim PRIVATE -Ofast -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes -DBROAD_W=16)

# # 调试版本
# add_executable(npusim_debug ${SRC_FILES})
# # set_target_properties(npusim_debug PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
# # target_link_libraries(npusim_debug systemc sfml-graphics sfml-system sfml-window m)
# target_link_libraries(npusim_debug
#     PUBLIC
#     systemc cairo sfml-graphics sfml-system sfml-window m pthread
#         DRAMSys::libdramsys
#         DRAMSys::util
#         DRAMSys::config
# )
# target_compile_options(npusim_debug PRIVATE -g -Ofast -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes -DBROAD_W=16)

# 清理目标的功能
# set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "npusim;npusim_debug")
set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "npusim;npusim_debug;load_config;load_config_debug")# add_library(systemc INTERFACE)

# add_library(SystemC::systemc ALIAS systemc)
