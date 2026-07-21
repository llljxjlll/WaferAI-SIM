#pragma once
#include <queue>
#include <vector>

#include "common/msg.h"
#include "monitor/config_helper_base.h"
#include "monitor/host_envelope.h"

using namespace std;

class config_helper_core : public config_helper_base {
public:
    int batch_size;
    int seq_len;

    config_helper_core(string filename,
                       int config_chip_id = 0);

    config_helper_core *clone() const override {
        return new config_helper_core(*this);
    }

    void generate_prims(int i);
    void calculate_address(bool do_loop);

    void printSelf();
    void random_core();

    void parse_ack_msg(Event_engine *event_engine, int flow_id,
                       sc_event *notify_event);
    void parse_done_msg(Event_engine *event_engine, sc_event *notify_event);

    void fill_queue_start(queue<Msg> *q);
    void fill_queue_config(queue<Msg> *q);

    // 2B0：信封接口——只决定「发给哪个全局核 + 什么 Msg」，不含物理 HOST lane。
    // fill_queue_* 现内部 = Build*Messages() + LegacyHostEnqueue（die0，逐位不变）；
    // 2B1 会改由 per-die HostAttachment 消费这些信封。
    std::vector<HostEnvelope> BuildConfigMessages();
    std::vector<HostEnvelope> BuildStartMessages();

    CoreConfig *get_core(int id);
};