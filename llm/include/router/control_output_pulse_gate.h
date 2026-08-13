#pragma once

#include <cstddef>
#include <stdexcept>

class ControlOutputPulseGate {
public:
    bool BeginCycle() noexcept {
        if (!cooldown_) return true;
        cooldown_ = false;
        return false;
    }

    void MarkSent() {
        if (cooldown_)
            throw std::logic_error(
                "control output sent without an observable low cycle");
        cooldown_ = true;
    }

    bool CoolingDown() const noexcept { return cooldown_; }
    std::size_t Residual() const noexcept { return cooldown_ ? 1U : 0U; }

private:
    bool cooldown_ = false;
};
