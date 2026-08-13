#pragma once
#include "common/msg.h"
#include "dte/sync_runtime.h"

// 消息（数据包）的工具函数
sc_bv<256> SerializeMsg(Msg msg);
Msg DeserializeMsg(sc_bv<256> buffer);

Msg MakeEventControlMsg(const EventControlMessage &event);
EventControlMessage ParseEventControlMsg(const Msg &msg);

// 给定计算原语的output大小，计算需要发送的包数量
void CalculatePacketNum(int output_size, int weight, int data_byte,
                        int &packet_num, int &end_length, int &packet_scale,
                        int &packets_in_last_group);
bool IsBlockableMsgType(MSG_TYPE type);