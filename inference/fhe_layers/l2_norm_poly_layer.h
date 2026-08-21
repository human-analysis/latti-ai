/*
 * Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "data_structs/feature0d.h"
#include "layer.h"

#include <vector>

class L2NormPolyLayer : public Layer {
public:
    L2NormPolyLayer(const ls::CkksParameter& param_in,
                    uint32_t n_channel_per_ct_in,
                    uint32_t input_level_in,
                    uint32_t output_level_in,
                    double a_in,
                    double b_in,
                    double c_in);

    void prepare_weight();

    Feature0DEncrypted run(ls::CkksContext& ctx, const Feature0DEncrypted& x) const;

    std::vector<ls::CkksPlaintextMul> a_pt;
    std::vector<ls::CkksPlaintextMul> b_pt;
    std::vector<ls::CkksPlaintext> c_pt;

private:
    ls::CkksCiphertext homomorphic_sum_packed_ct(ls::CkksContext& ctx, const ls::CkksCiphertext& x) const;

    uint32_t n_channel_per_ct_;
    uint32_t input_level_;
    uint32_t output_level_;
    double a_;
    double b_;
    double c_;
};
