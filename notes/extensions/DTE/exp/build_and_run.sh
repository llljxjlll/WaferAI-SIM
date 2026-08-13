#!/usr/bin/env bash
# Compiles channel_sweep.cpp against the project's own dte_unit.cpp object
# file (already built under ../../../build by the main npusim CMake build,
# so this exercises the real, current DTE model) and runs it, producing
# results.csv next to this script.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
BUILD="$ROOT/build"
OBJ="$BUILD/CMakeFiles/npusim.dir/llm/src"
SYSTEMC_HOME=/opt/systemc-2.3.3

for f in "$OBJ/dte/dte_unit.cpp.o" "$OBJ/trace/Event_engine.cpp.o" "$OBJ/trace/Trace_event.cpp.o"; do
    if [ ! -f "$f" ]; then
        echo "missing $f -- build npusim first (cmake --build build --target npusim --parallel 2)" >&2
        exit 1
    fi
done

c++ -std=c++17 -O2 \
    -DDRAMSYS_RESOURCE_DIR="\"$ROOT/DRAMSys/configs\"" \
    -DL1CACHESIZE=4194304 -DL2CACHESIZE=15099494 \
    -DNPUSIM_SOURCE_ROOT="\"$ROOT\"" \
    -DSQLITE_ENABLE_RTREE -DSQLITE_OMIT_LOAD_EXTENSION \
    -I"$ROOT/llm/include" \
    -I"$ROOT/DRAMSys/src/libdramsys" \
    -I"$ROOT/DRAMSys/src/util" \
    -I"$ROOT/DRAMSys/lib/nlohmann_json/include" \
    -I"$ROOT/DRAMSys/src/configuration" \
    -I"$BUILD/_deps/sqlite3-src" \
    -I"$SYSTEMC_HOME/include" \
    -DBROAD_W=16 \
    "$HERE/channel_sweep.cpp" \
    "$OBJ/dte/dte_unit.cpp.o" \
    "$OBJ/trace/Event_engine.cpp.o" \
    "$OBJ/trace/Trace_event.cpp.o" \
    -L"$SYSTEMC_HOME/lib-linux64" -Wl,-rpath,"$SYSTEMC_HOME/lib-linux64" \
    -lsystemc -lpthread \
    -o "$HERE/channel_sweep"

"$HERE/channel_sweep" > "$HERE/results.csv"
echo "wrote $HERE/results.csv"
