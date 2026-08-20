if(NOT DEFINED NPUSIM OR NOT EXISTS "${NPUSIM}")
    message(FATAL_ERROR "NPUSIM must name the production executable")
endif()

set(expected_status
    "[PROGRAM_IO] phase=preflight mode=unresolved initializations=0 probes=0 checksum=unavailable pass=0")

function(require_preflight_failure case_name)
    execute_process(
        COMMAND "${NPUSIM}" ${ARGN}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr)
    string(CONCAT combined "${stdout}" "${stderr}")
    if(result EQUAL 0)
        message(FATAL_ERROR "${case_name}: npusim unexpectedly succeeded")
    endif()
    string(FIND "${combined}" "${expected_status}" status_index)
    if(status_index EQUAL -1)
        message(FATAL_ERROR
            "${case_name}: missing machine-readable ProgramIo failure status\n${combined}")
    endif()
    string(FIND "${combined}" "Unknown option" unknown_index)
    if(NOT unknown_index EQUAL -1)
        message(FATAL_ERROR
            "${case_name}: ProgramIo flag hit a prefix-parser collision\n${combined}")
    endif()
endfunction()

require_preflight_failure(
    "paired flags"
    --program missing-program-io-artifact.npup
    --linked-manifest missing-program-io-manifest.json)

require_preflight_failure(
    "legacy probe exclusivity"
    --program missing-program-io-artifact.npup
    --linked-manifest missing-program-io-manifest.json
    --program-io missing-program-io-sidecar.json
    --p5-memory-probe missing-p5-sidecar.json)

message(STATUS "ProgramIo production CLI preflight: PASS")
