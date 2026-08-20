#pragma once

#include "isa/record_codec.h"
#include "prims/base.h"

inline constexpr uint8_t kExactStage2PrimWireVersion = 1;

class Exact_stage2_prim_base : public NpuBase {
public:
    void initialize() override;
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops,
                  u_int64_t &sfu_ops, u_int64_t &vec_ops) override;
    vector<sc_bv<128>> serialize() override;
    void deserialize(vector<sc_bv<128>> wire) override;
    void printSelf() override;

    Opcode exact_opcode() const noexcept { return exact_opcode_; }

protected:
    Exact_stage2_prim_base(const char *factory_name, Opcode opcode);
    virtual ExternalRecord ExactRecord() const = 0;
    virtual void AssignExactRecord(const ExternalRecord &record) = 0;

private:
    Opcode exact_opcode_;
};

class Rope_qk_exact_prim final : public Exact_stage2_prim_base {
public:
    RopeQkExactOperands operands;

    Rope_qk_exact_prim();

private:
    ExternalRecord ExactRecord() const override;
    void AssignExactRecord(const ExternalRecord &record) override;
};

class Attention_exact_prim final : public Exact_stage2_prim_base {
public:
    AttentionExactOperands operands;

    Attention_exact_prim();

private:
    ExternalRecord ExactRecord() const override;
    void AssignExactRecord(const ExternalRecord &record) override;
};

class Embedding_lookup_prim final : public Exact_stage2_prim_base {
public:
    EmbeddingLookupOperands operands;

    Embedding_lookup_prim();

private:
    ExternalRecord ExactRecord() const override;
    void AssignExactRecord(const ExternalRecord &record) override;
};

class Greedy_sample_prim final : public Exact_stage2_prim_base {
public:
    GreedySampleOperands operands;

    Greedy_sample_prim();

private:
    ExternalRecord ExactRecord() const override;
    void AssignExactRecord(const ExternalRecord &record) override;
};

class Cross_entropy_forward_prim final : public Exact_stage2_prim_base {
public:
    CrossEntropyForwardOperands operands;

    Cross_entropy_forward_prim();
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops,
                  u_int64_t &sfu_ops, u_int64_t &vec_ops) override;

private:
    ExternalRecord ExactRecord() const override;
    void AssignExactRecord(const ExternalRecord &record) override;
};

class Cross_entropy_backward_prim final : public Exact_stage2_prim_base {
public:
    CrossEntropyBackwardOperands operands;

    Cross_entropy_backward_prim();
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops,
                  u_int64_t &sfu_ops, u_int64_t &vec_ops) override;

private:
    ExternalRecord ExactRecord() const override;
    void AssignExactRecord(const ExternalRecord &record) override;
};

class Sgd_update_prim final : public Exact_stage2_prim_base {
public:
    SgdUpdateOperands operands;

    Sgd_update_prim();
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops,
                  u_int64_t &sfu_ops, u_int64_t &vec_ops) override;

private:
    ExternalRecord ExactRecord() const override;
    void AssignExactRecord(const ExternalRecord &record) override;
};
