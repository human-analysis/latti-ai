#include "l2_norm_poly_layer.h"

#include <cmath>
#include <utility>

using namespace std;

L2NormPolyLayer::L2NormPolyLayer(const ls::CkksParameter& param_in,
                                 uint32_t n_channel_per_ct_in,
                                 uint32_t input_level_in,
                                 uint32_t output_level_in,
                                 double a_in,
                                 double b_in,
                                 double c_in)
    : Layer(param_in),
      n_channel_per_ct_(n_channel_per_ct_in),
      input_level_(input_level_in),
      output_level_(output_level_in),
      a_(a_in),
      b_(b_in),
      c_(c_in) {}

void L2NormPolyLayer::prepare_weight() {
    ls::CkksContext ctx = ls::CkksContext::create_empty_context(param_);
    const int n_slots = param_.get_n() / 2;
    const double scale = param_.get_default_scale();

    a_pt.clear();
    b_pt.clear();
    c_pt.clear();

    // 4-level implementation:
    // x*x + rescale → level L-1, scale Δ
    // y*y + rescale → level L-2, scale Δ
    // mult_plain_mul(y², a_pt) + rescale → level L-3, scale Δ   (a_pt at L-2)
    // mult_plain_mul(y,  b_pt) + rescale → level L-3, scale Δ   (b_pt at L-2)
    // poly = a_term + b_term + c_pt at level L-3, scale Δ        (c_pt at L-3)
    // mult(x, poly) + rescale → level L-4, scale Δ               (output at Δ)
    a_pt.push_back(ctx.encode_mul(vector<double>(n_slots, a_), input_level_ - 2, scale));
    b_pt.push_back(ctx.encode_mul(vector<double>(n_slots, b_), input_level_ - 2, scale));
    c_pt.push_back(ctx.encode(vector<double>(n_slots, c_), input_level_ - 3, scale));
}

ls::CkksCiphertext L2NormPolyLayer::homomorphic_sum_packed_ct(ls::CkksContext& ctx,
                                                              const ls::CkksCiphertext& x) const {
    auto sum = x.copy();
    const int n_steps = (int)ceil(log2((double)n_channel_per_ct_));
    for (int i = 0; i < n_steps; i++) {
        auto rotated = ctx.rotate(sum, (1 << i));
        sum = ctx.add(sum, rotated);
    }
    return sum;
}

Feature0DEncrypted L2NormPolyLayer::run(ls::CkksContext& ctx, const Feature0DEncrypted& x) const {
    const double default_scale = param_.get_default_scale();

    // Step 1: norm_sq = sum(x²)  — 1 rescale → level input_level-1, scale Δ
    vector<ls::CkksCiphertext> partial_sums;
    partial_sums.reserve(x.data.size());
    for (const auto& x_ct : x.data) {
        auto x_sq = ctx.rescale(ctx.relinearize(ctx.mult(x_ct, x_ct)), default_scale);
        auto sum_ct = x_sq.copy();
        const int n_steps = (int)ceil(log2((double)x.n_channel_per_ct));
        for (int i = 0; i < n_steps; i++) {
            auto rotated = ctx.rotate(sum_ct, (1 << i) * (int)x.skip);
            sum_ct = ctx.add(sum_ct, rotated);
        }
        partial_sums.push_back(move(sum_ct));
    }

    auto norm_sq = partial_sums.at(0).copy();
    for (size_t i = 1; i < partial_sums.size(); i++) {
        auto partial = partial_sums[i].copy();
        while (norm_sq.get_level() > partial.get_level())
            norm_sq = ctx.drop_level(norm_sq);
        while (partial.get_level() > norm_sq.get_level())
            partial = ctx.drop_level(partial);
        norm_sq = ctx.add(norm_sq, partial);
    }
    // norm_sq at level input_level-1, scale Δ

    // Step 2: norm_sq_sq = y²  — 1 rescale → level input_level-2, scale Δ
    auto norm_sq_sq = ctx.rescale(ctx.relinearize(ctx.mult(norm_sq, norm_sq)), default_scale);

    // Align norm_sq down to match norm_sq_sq level
    auto norm_sq_for_b = norm_sq.copy();
    while (norm_sq_for_b.get_level() > norm_sq_sq.get_level())
        norm_sq_for_b = ctx.drop_level(norm_sq_for_b);

    // Step 3: poly = a·y² + b·y + c  — rescale after each pt-ct mult → poly at scale Δ
    // norm_sq_sq and norm_sq_for_b are both at L-2; a_pt and b_pt are encoded at L-2.
    auto a_term = ctx.rescale(ctx.mult_plain_mul(norm_sq_sq, a_pt.at(0)), default_scale);    // level L-3, scale Δ
    auto b_term = ctx.rescale(ctx.mult_plain_mul(norm_sq_for_b, b_pt.at(0)), default_scale); // level L-3, scale Δ
    auto poly = ctx.add(a_term, b_term);
    poly = ctx.add_plain(poly, c_pt.at(0));  // c_pt at level L-3, scale Δ
    // poly at level input_level-3, scale Δ

    // Step 4: x · poly + rescale  — 1 rescale → level input_level-4, scale Δ (standard)
    Feature0DEncrypted result(x.context, output_level_);
    result.n_channel = x.n_channel;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.skip = x.skip;
    result.data.reserve(x.data.size());

    for (const auto& x_ct : x.data) {
        auto x_for_mul = x_ct.copy();
        auto norm_for_mul = poly.copy();
        while (x_for_mul.get_level() > norm_for_mul.get_level())
            x_for_mul = ctx.drop_level(x_for_mul);
        while (norm_for_mul.get_level() > x_for_mul.get_level())
            norm_for_mul = ctx.drop_level(norm_for_mul);
        auto y_ct = ctx.rescale(ctx.relinearize(ctx.mult(x_for_mul, norm_for_mul)), default_scale);
        while (y_ct.get_level() > (int)output_level_)
            y_ct = ctx.drop_level(y_ct);
        result.data.push_back(move(y_ct));
    }

    return result;
}
