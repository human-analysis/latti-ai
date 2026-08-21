/*
 * Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "layer.h"
#include "../data_structs/feature2d.h"
#include <vector>

class HerPNLayer : public Layer {
public:
    HerPNLayer(const ls::CkksParameter& param_in,
               const Duo& input_shape_in,
               const Array<double, 2>& weight_in,
               const Duo& skip_in,
               uint32_t n_channel_per_ct_in,
               uint32_t level_in,
               const Duo& zero_skip_in = {1, 1},
               const Duo& block_expansion_in = {1, 1},
               bool is_ordinary_pack_in = false,
               bool deferred_rescale_in = false);

    void prepare_weight();
    void prepare_weight_lazy();

    ls::CkksPlaintext generate_const_plain(ls::CkksContext& ctx, int ct_idx, int level, double scale = 0.0) const;
    ls::CkksPlaintextMul generate_linear_mul(ls::CkksContext& ctx, int ct_idx, int level, double scale = 0.0) const;
    ls::CkksPlaintextMul generate_square_scale_mul(ls::CkksContext& ctx, int level) const;

    std::vector<ls::CkksCiphertext> run_core(ls::CkksContext& ctx, const std::vector<ls::CkksCiphertext>& x);
    Feature2DEncrypted run(ls::CkksContext& ctx, const Feature2DEncrypted& x);
    Array<double, 3> run_plaintext(const Array<double, 3>& x);

    Duo input_shape;
    Duo skip;
    Duo pre_skip;
    Duo block_expansion;
    Duo block_shape;
    Duo zero_skip;
    Array<double, 2> weight;
    uint32_t n_channel_per_ct;
    uint32_t n_block_per_ct;
    bool is_ordinary_pack;
    bool deferred_rescale;

    std::vector<ls::CkksPlaintext> const_plain_pt;
    std::vector<ls::CkksPlaintextMul> linear_mul_pt;
    std::vector<ls::CkksPlaintextMul> square_scale_mul_pt;

private:
    std::vector<double> generate_weight_values(int coeff_idx, int ct_idx) const;

    int N;
    int cached_channel;
    int cached_skip_prod;
    int cached_total_block_size;
};
