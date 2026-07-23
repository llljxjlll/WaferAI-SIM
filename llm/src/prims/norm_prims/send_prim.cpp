#include "systemc.h"

#include "defs/enums.h"
#include "prims/base.h"
#include "prims/norm_prims.h"
#include "utils/prim_utils.h"
#include "utils/print_utils.h"

REGISTER_PRIM(Send_prim);

void Send_prim::printSelf() {}

void Send_prim::deserialize(vector<sc_bv<128>> segments) {
    // d2d_exit_* 是一次执行期间的运行态，不能从上一次反序列化/执行继承。
    d2d_exit_port = -1;
    d2d_exit_selected = false;
    stripe_packets.clear();
    stripe_sent.clear();
    stripe_exit_ports.clear();
    next_subflow = 0;

    auto buffer = segments[0];

    des_id = buffer.range(23, 8).to_uint64();
    type = SEND_TYPE(buffer.range(59, 56).to_uint64());

    if (type == SEND_DATA)
        output_label =
            g_addr_label_table.findRecord(buffer.range(35, 24).to_uint64());

    max_packet = buffer.range(91, 60).to_uint64();
    tag_id = buffer.range(111, 92).to_uint64();
    end_length = buffer.range(119, 112).to_uint64();
    datatype = DATATYPE(buffer.range(121, 120).to_uint64());
    stripe_count = buffer.range(124, 122).to_uint64();
    if (stripe_count == 0)
        stripe_count = 1;
    if (stripe_count != 1 && stripe_count != 2 && stripe_count != 4)
        throw std::runtime_error("Send_prim stripe_count must be 1, 2, or 4");
}

vector<sc_bv<128>> Send_prim::serialize() {
    vector<sc_bv<128>> segments;

    sc_bv<128> d;
    d.range(7, 0) = sc_bv<8>(PrimFactory::getInstance().getPrimId(name));
    d.range(23, 8) = sc_bv<16>(des_id);

    if (type == SEND_DATA) {
        if (output_label == UNSET_LABEL) {
            LOG_ERROR(send_prim.cpp)
                << "SEND_DATA must have a set output_label";
        }
        
        d.range(35, 24) = sc_bv<12>(g_addr_label_table.addRecord(output_label));
    }

    d.range(59, 56) = sc_bv<4>(type);
    d.range(91, 60) = sc_bv<32>(max_packet);
    d.range(111, 92) = sc_bv<20>(tag_id);
    d.range(119, 112) = sc_bv<8>(end_length);
    d.range(121, 120) = sc_bv<2>(datatype);
    if (stripe_count != 1 && stripe_count != 2 && stripe_count != 4)
        throw std::runtime_error("Send_prim stripe_count must be 1, 2, or 4");
    d.range(124, 122) = sc_bv<3>(stripe_count);
    segments.push_back(d);

    return segments;
}
int Send_prim::taskCoreDefault(TaskCoreContext &context) {
#if USE_NB_DRAMSYS == 0
    auto wc = context.wc;
#endif
    auto mau = context.mau;
    auto hmau = context.hmau;
    sc_bv<128> msg_data;
    sc_time elapsed_time;

    // 找到output_label对应的数据块
    if (type == SEND_DATA) {
        bool need_delete = false;

        std::size_t pos = output_label.find("DEL_");
        if (pos != std::string::npos) {
            output_label = output_label.substr(pos + 4);
            need_delete = true;
        }

        AddrPosKey sc_key;
        int flag =
            prim_context->sram_pos_locator_->findPair(output_label, sc_key);
        sc_key.pos = 0;

#if USE_SRAM_MANAGER == 1
        mau->mem_read_port->read(0, msg_data, elapsed_time);
#else
        // ERROT SRAM BITWIDTH
        for (int i = 0; i < 1; i++) {
            wait(CYCLE, SC_NS);
        }
#endif
        if (need_delete)
            prim_context->sram_pos_locator_->deletePair(output_label);
    }

    msg_data = 0b1;
    return 0;
}