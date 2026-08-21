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

#include "herpn_layer.h"

#include <cmath>
#include <stdexcept>

using namespace std;
using namespace cxx_sdk_v2;

HerPNLayer::HerPNLayer(const CkksParameter& param_in,
                       const Duo& input_shape_in,
                       const Array<double, 2>& weight_in,
                       const Duo& skip_in,
                       uint32_t n_channel_per_ct_in,
                       uint32_t level_in,
                       const Duo& zero_skip_in,
                       const Duo& block_expansion_in,
                       bool is_ordinary_pack_in,
                       bool deferred_rescale_in)
    : Layer(param_in), input_shape(input_shape_in), skip(skip_in), weight(weight_in.copy()) {
    if ((input_shape[0] & (input_shape[0] - 1)) != 0 || (input_shape[1] & (input_shape[1] - 1)) != 0) {
        throw invalid_argument("input_shape must be powers of 2, got: [" + to_string(input_shape[0]) + ", " +
                               to_string(input_shape[1]) + "]");
    }
    if ((skip[0] & (skip[0] - 1)) != 0 || (skip[1] & (skip[1] - 1)) != 0) {
        throw invalid_argument("skip must be powers of 2, got: [" + to_string(skip[0]) + ", " + to_string(skip[1]) +
                               "]");
    }

    level_ = level_in;
    n_channel_per_ct = n_channel_per_ct_in;
    zero_skip = zero_skip_in;
    block_expansion = block_expansion_in;
    pre_skip[0] = skip[0] * zero_skip[0];
    pre_skip[1] = skip[1] * zero_skip[1];
    n_block_per_ct = ceil(n_channel_per_ct / double(skip[0] * skip[1]));
    block_shape[0] = input_shape[0] * skip[0] / block_expansion[0];
    block_shape[1] = input_shape[1] * skip[1] / block_expansion[1];
    if ((block_shape[0] & (block_shape[0] - 1)) != 0 || (block_shape[1] & (block_shape[1] - 1)) != 0) {
        throw invalid_argument("block_shape must be powers of 2, got: [" + to_string(block_shape[0]) + ", " +
                               to_string(block_shape[1]) + "]");
    }

    is_ordinary_pack = is_ordinary_pack_in;
    deferred_rescale = deferred_rescale_in;
    N = param_in.get_n();
    cached_channel = weight.get_shape()[1];
    cached_skip_prod = skip[0] * skip[1];
    cached_total_block_size = n_block_per_ct * block_shape[0] * block_shape[1];
}

vector<double> HerPNLayer::generate_weight_values(int coeff_idx, int ct_idx) const {
    vector<double> feature_tmp_pack(N / 2, 0.0);
    if (is_ordinary_pack) {
        for (int ch = 0; ch < (int)n_channel_per_ct; ch++) {
            int channel_idx = ct_idx * n_channel_per_ct + ch;
            if (channel_idx >= cached_channel)
                continue;
            for (int j = 0; j < input_shape[0]; j++) {
                for (int k = 0; k < input_shape[1]; k++) {
                    int index = ch * block_shape[0] * block_shape[1] + j * block_shape[1] * skip[0] + k * skip[1];
                    feature_tmp_pack[index] = weight.get(coeff_idx, channel_idx);
                }
            }
        }
    } else {
        for (int linear_idx = 0; linear_idx < cached_total_block_size; ++linear_idx) {
            int block_i = linear_idx / (block_shape[0] * block_shape[1]);
            int residual = linear_idx % (block_shape[0] * block_shape[1]);
            int shape_i = residual / block_shape[1];
            int shape_j = residual % block_shape[1];

            int channel_idx = ct_idx * n_channel_per_ct / block_expansion[0] / block_expansion[1] +
                              block_i * cached_skip_prod + (skip[0] * (shape_i % skip[0]) + shape_j % skip[0]);
            if (channel_idx >= cached_channel || (shape_i % pre_skip[0]) >= skip[0] ||
                (shape_j % pre_skip[1]) >= skip[1])
                continue;

            int index = block_i * block_shape[0] * block_shape[1] + shape_i * block_shape[1] + shape_j;
            feature_tmp_pack[index] = weight.get(coeff_idx, channel_idx);
        }
    }
    return feature_tmp_pack;
}

void HerPNLayer::prepare_weight() {
    int n_packed_out_channel = div_ceil(cached_channel, n_channel_per_ct) * block_expansion[0] * block_expansion[1];
    const_plain_pt.clear();
    linear_mul_pt.clear();
    square_scale_mul_pt.clear();
    const_plain_pt.resize(n_packed_out_channel);
    linear_mul_pt.resize(n_packed_out_channel);

    CkksContext ctx = CkksContext::create_empty_context(this->param_);
    const double default_scale = param_.get_default_scale();
    const double rescale_ratio = param_.get_q(level_) / default_scale;
    const bool correct_square_scale = !deferred_rescale && (rescale_ratio > 2.0 || rescale_ratio < 0.5);
    if (correct_square_scale)
        square_scale_mul_pt.resize(n_packed_out_channel);
    ctx.resize_copies(n_packed_out_channel);
    parallel_for(n_packed_out_channel, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        const auto c0 = generate_weight_values(0, ct_idx);
        const auto c1 = generate_weight_values(1, ct_idx);
        const double default_scale = ctx_copy.get_parameter().get_default_scale();
        const int const_level = deferred_rescale ? level_ : level_ - 1;
        const double const_scale = deferred_rescale ? default_scale * default_scale : default_scale;
        const_plain_pt[ct_idx] = ctx_copy.encode(c0, const_level, const_scale);
        const double linear_scale = deferred_rescale ? default_scale : param_.get_q(level_);
        linear_mul_pt[ct_idx] = ctx_copy.encode_mul(c1, level_, linear_scale);
        if (correct_square_scale)
            square_scale_mul_pt[ct_idx] = generate_square_scale_mul(ctx_copy, level_);
    });
}

void HerPNLayer::prepare_weight_lazy() {
    const_plain_pt.clear();
    linear_mul_pt.clear();
    square_scale_mul_pt.clear();
}

CkksPlaintext HerPNLayer::generate_const_plain(CkksContext& ctx, int ct_idx, int level, double scale) const {
    const auto c0 = generate_weight_values(0, ct_idx);
    double s = (scale > 0.0) ? scale : ctx.get_parameter().get_default_scale();
    return ctx.encode(c0, level, s);
}

CkksPlaintextMul HerPNLayer::generate_linear_mul(CkksContext& ctx, int ct_idx, int level, double scale) const {
    const auto c1 = generate_weight_values(1, ct_idx);
    double s = (scale > 0.0) ? scale : ctx.get_parameter().get_default_scale();
    return ctx.encode_mul(c1, level, s);
}

CkksPlaintextMul HerPNLayer::generate_square_scale_mul(CkksContext& ctx, int level) const {
    const double correction_scale = ctx.get_parameter().get_q(level) /
                                    ctx.get_parameter().get_default_scale();
    return ctx.encode_mul(vector<double>(N / 2, 1.0), level, correction_scale);
}

vector<CkksCiphertext> HerPNLayer::run_core(CkksContext& ctx, const vector<CkksCiphertext>& x) {
    vector<CkksCiphertext> result(x.size());
    parallel_for(x.size(), th_nums, ctx, [&](CkksContext& ctx_copy, int x_idx) {
        // Use the actual input scale for coefficient encoding so square and linear
        // terms end up at the same scale after rescaling (e.g. input at 2^60 →
        // square at 2^74, linear needs c1 encoded at 2^60 to also rescale to 2^74).
        double input_scale = x[x_idx].get_scale();

        auto square_raw = ctx_copy.relinearize(ctx_copy.mult(x[x_idx], x[x_idx]));
        const double rescale_ratio = ctx_copy.get_parameter().get_q(x[x_idx].get_level()) /
                                     ctx_copy.get_parameter().get_default_scale();
        const bool correct_square_scale =
            !deferred_rescale && (rescale_ratio > 2.0 || rescale_ratio < 0.5);
        if (correct_square_scale) {
            auto square_scale_pt = generate_square_scale_mul(ctx_copy, x[x_idx].get_level());
            square_raw = ctx_copy.mult_plain_mul(square_raw, square_scale_pt);
        }
        auto square = deferred_rescale
                          ? std::move(square_raw)
                          : ctx_copy.rescale(square_raw, ctx_copy.get_parameter().get_default_scale());
        double output_scale = square.get_scale();

        const double linear_scale = deferred_rescale ? input_scale : ctx_copy.get_parameter().get_q(x[x_idx].get_level());
        auto c1_pt = generate_linear_mul(ctx_copy, x_idx, x[x_idx].get_level(), linear_scale);
        auto linear_raw = ctx_copy.mult_plain_mul(x[x_idx], c1_pt);
        auto linear = deferred_rescale
                          ? std::move(linear_raw)
                          : ctx_copy.rescale(linear_raw, ctx_copy.get_parameter().get_default_scale());

        result[x_idx] = ctx_copy.add(square, linear);

        auto c0_pt = generate_const_plain(ctx_copy, x_idx, result[x_idx].get_level(), output_scale);
        result[x_idx] = ctx_copy.add_plain(result[x_idx], c0_pt);
    });
    return result;
}

Feature2DEncrypted HerPNLayer::run(CkksContext& ctx, const Feature2DEncrypted& x) {
    Feature2DEncrypted result(&ctx, deferred_rescale ? x.level : x.level - 1);
    result.data = run_core(ctx, x.data);
    result.skip = x.skip;
    result.shape = x.shape;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.dim = x.dim;
    result.n_channel = x.n_channel;
    result.level = result.data.empty() ? (deferred_rescale ? x.level : x.level - 1) : result.data[0].get_level();
    return result;
}

Array<double, 3> HerPNLayer::run_plaintext(const Array<double, 3>& x) {
    int n_out_channel = x.get_shape()[0];
    Array<double, 3> result({x.get_shape()[0], x.get_shape()[1], x.get_shape()[2]});
    for (int in_channel_idx = 0; in_channel_idx < n_out_channel; in_channel_idx++) {
        for (int i = 0; i < input_shape[0]; i++) {
            for (int j = 0; j < input_shape[1]; j++) {
                if (i % zero_skip[0] == 0 && j % zero_skip[1] == 0) {
                    auto v = x.get(in_channel_idx, i, j);
                    result.set(in_channel_idx, i, j,
                               v * v + weight.get(1, in_channel_idx) * v + weight.get(0, in_channel_idx));
                }
            }
        }
    }
    return result;
}
