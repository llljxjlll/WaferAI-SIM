#include "monitor/workload_rendezvous_selftest.h"
#include "monitor/workload_normalize.h"
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
WLJson ValidWorkload() {
    return WLJson::parse(R"JSON(
{
  "source": [{"dest": 0, "size": 128}],
  "chips": [{"chip_id": 0, "cores": [
    {"id": 0, "worklist": [
      {"recv_cnt": 1, "cast": [{"dest": 1, "tag": 41}]}
    ]},
    {"id": 1, "worklist": [
      {"recv_cnt": 0, "cast": []},
      {"recv_cnt": 1, "recv_tag": 41, "cast": [{"dest": -1}]}
    ]}
  ]}]
}
)JSON");
}

bool ThrowsWith(const WLJson &workload, const std::string &needle) {
    try {
        ValidateWorkloadRendezvous(workload, 0);
    } catch (const std::exception &error) {
        return std::string(error.what()).find(needle) != std::string::npos;
    }
    return false;
}
} // namespace

int RunWorkloadRendezvousSelfTest() {
    int failures = 0;
    auto check = [&](bool condition, const char *name) {
        if (!condition) {
            ++failures;
            std::cerr << "[WORKLOAD_RENDEZVOUS] FAIL " << name << '\n';
        }
    };

    const WLJson valid = ValidWorkload();
    try {
        ValidateWorkloadRendezvous(valid, 0);
        std::cout << "[WORKLOAD_RENDEZVOUS] PASS valid producer graph\n";
    } catch (const std::exception &error) {
        std::cerr << "[WORKLOAD_RENDEZVOUS] FAIL valid producer graph: "
                  << error.what() << '\n';
        ++failures;
    }

    WLJson no_root = valid;
    no_root["source"] = WLJson::array();
    no_root["chips"][0]["cores"][1]["worklist"].erase(0);
    no_root["chips"][0]["cores"][1]["worklist"][0]["cast"] =
        WLJson::array({{{"dest", 0}, {"tag", 0}}});
    check(ThrowsWith(no_root, "no runnable root"),
          "source-free rendezvous cycle");

    WLJson missing_start = valid;
    missing_start["chips"][0]["cores"][0]["worklist"][0]["recv_cnt"] = 2;
    check(ThrowsWith(missing_start, "RECV_START expects"),
          "unsatisfied RECV_START");

    WLJson missing_producer = valid;
    missing_producer["chips"][0]["cores"][0]["worklist"][0]["cast"] =
        WLJson::array();
    check(ThrowsWith(missing_producer, "declared producer"),
          "unsatisfied recv tag");

    WLJson duplicate_core = valid;
    duplicate_core["chips"][0]["cores"][1]["id"] = 0;
    check(ThrowsWith(duplicate_core, "duplicate workload core id"),
          "duplicate core id");

    WLJson bad_source = valid;
    bad_source["source"][0]["dest"] = 9;
    check(ThrowsWith(bad_source, "active workload core"),
          "source to inactive core");

    if (failures == 0)
        std::cout << "[WORKLOAD_RENDEZVOUS] PASS all 6 cases\n";
    return failures;
}
