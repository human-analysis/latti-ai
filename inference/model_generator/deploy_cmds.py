# Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inference.lattisense.frontend.custom_task import *
from inference.model_generator.layers.activation_layer import *
from inference.model_generator.layers.add_pack import *
from inference.model_generator.layers.avgpool2d_layer import *
from inference.model_generator.layers.concat_layer import *
from inference.model_generator.layers.conv1d_packed_layer import *
from inference.model_generator.layers.conv2d_depthwise import *
from inference.model_generator.layers.conv2d_packed_layer import *
from inference.model_generator.layers.dense_packed_layer import *
from inference.model_generator.layers.inverse_multiplexed_conv2d_layer import *
from inference.model_generator.layers.inverse_multiplexed_depthwise_conv2d_layer import *
from inference.model_generator.layers.mult_scaler import *
from inference.model_generator.layers.multiplexed_conv1d_pack_layer import *
from inference.model_generator.layers.multiplexed_conv2d_pack_layer import *
from inference.model_generator.layers.multiplexed_conv2d_pack_layer_depthwise import *
from inference.model_generator.layers.poly_relu2d import *
from inference.model_generator.layers.upsample_layer import *
from training.model_compiler.components import (
    CryptoFaceNet4H192H32,
    N16QP1546H192H32,
    PN13QP218,
    PN14QP438,
    PN15QP880,
    PN16QP1761,
)


def _poly_bsgs_coeff_is_plain_add(order: int, idx: int) -> bool:
    baby_steps = math.ceil(math.sqrt(order + 1))
    giant_step = idx // baby_steps
    baby_step = idx % baby_steps
    if baby_step != 0:
        return False
    end_idx = min(giant_step * baby_steps + baby_steps - 1, order)
    return end_idx > idx


def _poly_bsgs_coeff_level(order: int, input_level: int, idx: int) -> int:
    baby_steps = math.ceil(math.sqrt(order + 1))
    giant_steps = math.ceil((order + 1) / baby_steps)

    powers: dict[int, tuple[int, int]] = {1: (0, input_level)}

    def compute_power(n: int) -> tuple[int, int]:
        if n in powers:
            return powers[n]
        best_depth = 10**9
        best_level = input_level
        for a in range(1, n // 2 + 1):
            b = n - a
            depth_a, level_a = compute_power(a)
            depth_b, level_b = compute_power(b)
            depth = max(depth_a, depth_b) + 1
            level = min(level_a, level_b) - 1
            if depth < best_depth:
                best_depth = depth
                best_level = level
        powers[n] = (best_depth, best_level)
        return powers[n]

    for n in range(2, order + 1):
        compute_power(n)

    required = set(range(1, baby_steps + 1))
    for g in range(1, giant_steps):
        giant_power = g * baby_steps
        if giant_power <= order:
            required.add(giant_power)

    max_depth = -1
    max_power_level = input_level
    for p in required:
        depth, power_level = powers[p]
        if depth > max_depth:
            max_depth = depth
            max_power_level = power_level

    bsgs_output_level = max_power_level - 1
    giant_step = idx // baby_steps
    baby_step = idx % baby_steps
    target_level = bsgs_output_level if giant_step == 0 else bsgs_output_level + 1
    return target_level if baby_step == 0 else target_level + 1


def read_config(config_path):
    with open(config_path, 'r', encoding='utf8') as fp:
        config_ctx = json.load(fp)
    return config_ctx


def _duo_value(node: dict, key: str, idx: int, fallback: int) -> int:
    if key not in node:
        return fallback
    return int(node[key][idx])


def normalize_flatten_metadata(config_info: dict) -> dict:
    if 'feature' not in config_info or 'layer' not in config_info:
        return config_info

    features = config_info['feature']
    for layer in config_info['layer'].values():
        layer_type = layer['type']
        input_id = layer['feature_input'][0]
        output_id = layer['feature_output'][0]
        if input_id not in features or output_id not in features:
            continue

        input_feature = features[input_id]
        output_feature = features[output_id]
        if layer_type == 'avgpool2d':
            if input_feature.get('dim') != 2 or output_feature.get('dim') != 2:
                continue
            stride0, stride1 = int(layer['stride'][0]), int(layer['stride'][1])
            output_feature['shape'] = [int(input_feature['shape'][0]) // stride0, int(input_feature['shape'][1]) // stride1]
            output_feature['skip'] = [int(input_feature['skip'][0]) * stride0, int(input_feature['skip'][1]) * stride1]
            if layer.get('is_adaptive_avgpool', False):
                inv0 = _duo_value(input_feature, 'invalid_fill', 0, 1)
                inv1 = _duo_value(input_feature, 'invalid_fill', 1, 1)
                output_feature['invalid_fill'] = [inv0 * stride0, inv1 * stride1]
        elif layer_type == 'reshape':
            if input_feature.get('dim') != 2 or output_feature.get('dim') != 0:
                continue
            inv0 = _duo_value(input_feature, 'invalid_fill', 0, 1)
            inv1 = _duo_value(input_feature, 'invalid_fill', 1, 1)
            output_feature['special_info'] = {
                'shape': [int(input_feature['shape'][0]), int(input_feature['shape'][1])],
                'skip': [int(input_feature['skip'][0]), int(input_feature['skip'][1])],
                'invalid_fill': [inv0, inv1],
            }
            output_feature['skip'] = (
                int(input_feature['shape'][0])
                * int(input_feature['skip'][0])
                * int(input_feature['shape'][1])
                * int(input_feature['skip'][1])
            )
    return config_info


_FHE_PARAMS = {
    'PN13QP218': PN13QP218,
    'PN14QP438': PN14QP438,
    'PN15QP880': PN15QP880,
    'PN16QP1761': PN16QP1761,
    'N16QP1546H192H32': N16QP1546H192H32,
    'CryptoFaceNet4H192H32': CryptoFaceNet4H192H32,
}


def set_param(param_name):
    if param_name not in _FHE_PARAMS:
        raise ValueError(f'Unsupported FHE parameter name: {param_name!r}')
    fhe = _FHE_PARAMS[param_name]
    if param_name == 'CryptoFaceNet4H192H32':
        param = CkksBtpParam.create_cryptoface_net4_param()
    elif param_name == 'N16QP1546H192H32':
        param = CkksBtpParam.create_default_param()
    else:
        param = CkksParam.create_custom_param(n=fhe.poly_modulus_degree, p=fhe.p, q=fhe.q)
    set_fhe_param(param)


def gen_custom_task(task_path, param_name='PN14QP438', use_gpu=True, style='ordinary'):
    n = _FHE_PARAMS[param_name].poly_modulus_degree
    fhe = _FHE_PARAMS[param_name]
    set_param(param_name)
    task_config_info = read_config(os.path.join(task_path, 'task_config.json'))
    try:
        block_shape = task_config_info['block_shape']
    except Exception:
        block_shape = [64, 64]
    config_info = normalize_flatten_metadata(read_config(os.path.join(task_path, 'nn_layers_ct_0.json')))

    # Re-select BTP param based on n_mult_level in the compiled ckks_parameter.json.
    # set_param() above picks create_default_param() (btp_output_level=9) for n=65536,
    # but CryptoFaceNet4 compilations use n_mult_level=16 (btp_output_level=16).
    if n == 65536:
        ckks_param_path = os.path.join(task_path, 'ckks_parameter.json')
        if os.path.exists(ckks_param_path):
            ckks_cfg = read_config(ckks_param_path)
            compiled_n_mult_level = next(iter(ckks_cfg.values())).get('n_mult_level', 9)
            if compiled_n_mult_level == 16:
                set_fhe_param(CkksBtpParam.create_cryptoface_net4_param())

    input_args = list()
    feature_id_to_nodes_map = {}
    task_output_feature_ids = config_info['output_feature']

    # Pre-add all graph-level input ciphertexts first so they precede weights in input_args.
    # This ensures the C++ signature position-matching works correctly for multi-input models.
    # Only needed when there are multiple graph-level inputs; single-input models are handled
    # correctly by the lazy-loading in the main loop (which also computes the right CT count
    # for big_size layers).
    if len(config_info['input_feature']) > 1:
        for input_fid in config_info['input_feature']:
            feat = config_info['feature'][input_fid]
            pack = int(feat['pack_num'])
            level = int(feat['level'])
            n_packed = math.ceil(int(feat['channel']) / pack)
            x = [CkksCiphertextNode(input_fid + f'input{k}', level=level) for k in range(n_packed)]
            feature_id_to_nodes_map[input_fid] = x
            input_args.append(Argument(input_fid, x))

    for layer_id, layer_config in config_info['layer'].items():
        if 'relu2d' in layer_config['type'] or 'split' == layer_config['type']:
            continue
        layer_input_feature_ids = layer_config['feature_input']
        layer_output_feature_ids = layer_config['feature_output']
        groups = 1
        n_in_channel = int(layer_config['channel_input'])
        n_out_channel = int(layer_config['channel_output'])

        skip = config_info['feature'][layer_input_feature_ids[0]]['skip']
        pack = int(config_info['feature'][layer_input_feature_ids[0]]['pack_num'])
        level = int(config_info['feature'][layer_input_feature_ids[0]]['level'])
        n_packed_in_channel = math.ceil(n_in_channel / pack)
        n_packed_out_channel = math.ceil(n_out_channel / pack)
        if layer_config['type'] == 'fc0':
            n_packed_in_channel = math.ceil(n_in_channel / 8192)
            n_packed_out_channel = math.ceil(n_out_channel / pack)

        # For big_conv/big_avgpool/big_mult_scalar, input CT count differs from the default n_packed_in_channel
        if (
            layer_config['type'] == 'conv2d'
            or 'avgpool' in layer_config['type']
            or layer_config['type'] == 'mult_scalar'
        ) and layer_config.get('is_big_size', False):
            input_shape = config_info['feature'][layer_input_feature_ids[0]]['shape']
            block_expansion = (
                math.ceil(input_shape[0] / block_shape[0]),
                math.ceil(input_shape[1] / block_shape[1]),
            )
            n_packed_in_channel = n_in_channel * block_expansion[0] * block_expansion[1]

        for input_node in layer_input_feature_ids:
            if input_node not in feature_id_to_nodes_map.keys():
                x = [CkksCiphertextNode(input_node + f'input{k}', level=level) for k in range(n_packed_in_channel)]
                feature_id_to_nodes_map.update({input_node: x})
                input_args.append(Argument(input_node, x))

        if layer_config['type'] == 'reshape':
            layer_output_nodes = feature_id_to_nodes_map[layer_input_feature_ids[0]]
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif layer_config['type'] == 'conv2d':
            input_shape = config_info['feature'][layer_input_feature_ids[0]]['shape']
            groups = layer_config['groups']
            kernel_shape = layer_config['kernel_shape']
            stride = layer_config['stride']
            index = int(kernel_shape[0] * kernel_shape[1])
            is_big_conv = layer_config['is_big_size']
            block_expansion = (math.ceil(input_shape[0] / block_shape[0]), math.ceil(input_shape[1] / block_shape[1]))
            next_stride = [block_expansion[0] // stride[0], block_expansion[1] // stride[1]]
            padding = [-1, -1]
            if is_big_conv:
                if groups == n_out_channel and groups != 1:
                    big_conv = InverseMultiplexedDepthwiseConv2DLayer(
                        n_out_channel,
                        input_shape,
                        padding,
                        kernel_shape,
                        stride,
                        next_stride,
                        skip,
                        block_shape,
                    )

                    weight_pt = [
                        [
                            CkksPlaintextRingtNode(f'convw_{layer_id}_{k}_{i}')
                            for i in range(int(index * next_stride[0] * next_stride[1]))
                        ]
                        for k in range(n_out_channel)
                    ]
                else:
                    big_conv = InverseMultiplexedConv2DLayer(
                        n_out_channel,
                        n_in_channel,
                        input_shape,
                        padding,
                        kernel_shape,
                        stride,
                        next_stride,
                        skip,
                        block_shape,
                    )

                    weight_pt = [
                        [
                            [
                                CkksPlaintextRingtNode(f'convw_{layer_id}_{k}_{n}_{i}')
                                for i in range(int(index * next_stride[0] * next_stride[1]))
                            ]
                            for n in range(n_in_channel)
                        ]
                        for k in range(n_out_channel)
                    ]

                bias_pt = [CkksPlaintextRingtNode(f'convb_{layer_id}_{i}') for i in range(n_out_channel)]

                layer_output_nodes = big_conv.call(
                    feature_id_to_nodes_map[layer_input_feature_ids[0]], weight_pt, bias_pt, n
                )
                feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
                input_args.append(Argument(f'convw_{layer_id}', weight_pt))
                input_args.append(Argument(f'convb_{layer_id}', bias_pt))
            else:
                if style == 'ordinary':
                    if groups == n_out_channel and groups != 1:
                        conv0_layer = Conv2DPackedDepthwiseLayer(
                            n_out_channel,
                            n_in_channel,
                            input_shape,
                            kernel_shape,
                            stride,
                            skip,
                            pack,
                            n_packed_in_channel,
                            n_packed_out_channel,
                        )
                        weight_pt = [
                            [CkksPlaintextRingtNode(f'convw_{layer_id}_{n}_{i}') for i in range(index)]
                            for n in range(n_packed_out_channel)
                        ]
                    else:
                        conv0_layer = Conv2DPackedLayer(
                            n_out_channel,
                            n_in_channel,
                            input_shape,
                            kernel_shape,
                            stride,
                            skip,
                            pack,
                            n_packed_in_channel,
                            n_packed_out_channel,
                        )
                        weight_pt = [
                            [
                                [CkksPlaintextRingtNode(f'convw_{layer_id}_{n}_{m}_{i}') for i in range(index)]
                                for m in range(int(n_packed_in_channel * pack))
                            ]
                            for n in range(n_packed_out_channel)
                        ]

                    bias_pt = [CkksPlaintextRingtNode(f'convb_{layer_id}_{i}') for i in range(n_packed_out_channel)]
                    input_args.append(Argument(f'convw_{layer_id}', weight_pt))
                    input_args.append(Argument(f'convb_{layer_id}', bias_pt))
                    layer_output_nodes = conv0_layer.call(
                        feature_id_to_nodes_map[layer_input_feature_ids[0]], weight_pt, bias_pt
                    )
                    feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
                if style == 'multiplexed':
                    n_in_channel_per_ct = pack
                    n_block_per_ct = n_in_channel_per_ct // (skip[0] * skip[1])
                    n_out_channel_per_ct = n_in_channel_per_ct * stride[0] * stride[1]
                    n_pack_in_channel = math.ceil(n_in_channel / n_in_channel_per_ct)
                    n_pack_out_channel = math.ceil(n_out_channel / n_out_channel_per_ct)
                    kernel_size = kernel_shape[0] * kernel_shape[1]
                    if groups == n_out_channel and groups != 1:
                        conv0_layer = ParMultiplexedConv2DPackedLayerDepthwise(
                            n_out_channel,
                            n_in_channel,
                            input_shape,
                            kernel_shape,
                            stride,
                            skip,
                            n_in_channel_per_ct,
                            n_packed_in_channel,
                            n_packed_out_channel,
                        )
                        weight_pt = [
                            [CkksPlaintextRingtNode(f'convw_{layer_id}_{j}_{k}') for k in range(kernel_size)]
                            for j in range(n_pack_in_channel)
                        ]
                        bias_pt = [CkksPlaintextRingtNode(f'convb_{layer_id}_{i}') for i in range(n_pack_out_channel)]
                        if stride[0] != 1:
                            mask_pt = [CkksPlaintextRingtNode(f'convm_{layer_id}_{i}') for i in range(n_out_channel)]
                            input_args.append(Argument(f'convm_{layer_id}', mask_pt))
                        else:
                            mask_pt = []
                    else:
                        conv0_layer = ParMultiplexedConv2DPackedLayer(
                            n_out_channel,
                            n_in_channel,
                            input_shape,
                            kernel_shape,
                            stride,
                            skip,
                            pack,
                            n_packed_in_channel,
                            n_packed_out_channel,
                        )

                        use_folded_lc1 = layer_config.get('mux_conv_strategy') == 'folded_lc1'
                        size_0 = n_out_channel if use_folded_lc1 else math.ceil(n_out_channel / n_block_per_ct)
                        size_1 = n_pack_in_channel * n_block_per_ct

                        weight_pt = [
                            [
                                [CkksPlaintextRingtNode(f'convw_{layer_id}_{i}_{j}_{k}') for k in range(kernel_size)]
                                for j in range(size_1)
                            ]
                            for i in range(size_0)
                        ]
                        bias_pt = [CkksPlaintextRingtNode(f'convb_{layer_id}_{i}') for i in range(n_pack_out_channel)]
                        if use_folded_lc1 or (
                            stride[0] == 1 and stride[1] == 1 and skip[0] == 1 and skip[1] == 1
                        ):
                            mask_pt = []
                        else:
                            mask_size = min(n_block_per_ct, n_out_channel)
                            mask_pt = [
                                [CkksPlaintextRingtNode(f'convm_{layer_id}_{i}_{j}') for j in range(mask_size)]
                                for i in range(size_0)
                            ]
                            input_args.append(Argument(f'convm_{layer_id}', mask_pt))

                    input_args.append(Argument(f'convw_{layer_id}', weight_pt))
                    input_args.append(Argument(f'convb_{layer_id}', bias_pt))
                    if (
                        groups == 1
                        and not layer_config.get('is_big_size', False)
                        and layer_config.get('mux_conv_strategy') == 'folded_lc1'
                    ):
                        layer_output_nodes = conv0_layer.call_folded_lc1(
                            feature_id_to_nodes_map[layer_input_feature_ids[0]], weight_pt, bias_pt
                        )
                    else:
                        layer_output_nodes = conv0_layer.call(
                            feature_id_to_nodes_map[layer_input_feature_ids[0]], weight_pt, bias_pt, mask_pt
                        )
                    feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif layer_config['type'] in ('batchnorm2d', 'dropout', 'constmul', 'identity'):
            layer_output_nodes = feature_id_to_nodes_map[layer_input_feature_ids[0]]
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif 'square' in layer_config['type']:
            if 'square2d' in layer_config['type']:
                # square2d is treated as zero-level-cost in the compiled JSON:
                # the compiler defers the rescale to the next conv.  Use
                # mult_relin only (no rescale) so the output level stays the
                # same as the input, matching the JSON level assignment.
                input_nodes = feature_id_to_nodes_map[layer_input_feature_ids[0]]
                layer_output_nodes = [mult_relin(node, node) for node in input_nodes]
            else:
                continue
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif layer_config['type'] in ('poly_relu2d', 'simple_polyrelu'):
            feat = config_info['feature'][layer_input_feature_ids[0]]
            level = int(feat['level'])
            order = layer_config['order']
            n_pack_in_channel = math.ceil(n_in_channel / pack)
            if feat['dim'] == 0:
                input_shape = [1, 1]
                skip = [1, 1]
            else:
                input_shape = feat['shape']
            weight_pt = []
            for i in range(order + 1):
                node_cls = CkksPlaintextNode if _poly_bsgs_coeff_is_plain_add(order, i) else CkksPlaintextMulNode
                coeff_level = _poly_bsgs_coeff_level(order, level, i)
                weight_pt.append(
                    [node_cls(f'poly_reluw_{layer_id}_{i}_{j}', level=coeff_level) for j in range(n_pack_in_channel)]
                )
            polyrelu = PolyRelu(input_shape, order, skip, pack)
            feature_id_in_nodes = feature_id_to_nodes_map[layer_input_feature_ids[0]]
            drop_level_n = feature_id_in_nodes[0].level - level

            if level < feature_id_in_nodes[0].level:
                feature_id_in_nodes = [drop_level(node, drop_level_n) for node in feature_id_in_nodes]
            if feat['dim'] == 0:
                layer_output_nodes = polyrelu.call_bsgs_feature0d(feature_id_in_nodes, weight_pt)
            else:
                layer_output_nodes = polyrelu.call_bsgs_feature2d(feature_id_in_nodes, weight_pt)
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
            for i in range(len(weight_pt)):
                input_args.append(Argument(f'poly_reluw_{layer_id}_{i}', weight_pt[i]))

        elif layer_config['type'] == 'herpn2d':
            feat = config_info['feature'][layer_input_feature_ids[0]]
            level = int(feat['level'])
            n_pack_in_channel = math.ceil(n_in_channel / pack)
            input_nodes = feature_id_to_nodes_map[layer_input_feature_ids[0]]
            if level < input_nodes[0].level:
                input_nodes = [drop_level(node, input_nodes[0].level - level) for node in input_nodes]

            output_level = int(config_info['feature'][layer_output_feature_ids[0]]['level'])
            deferred_rescale = output_level == level
            rescale_ratio = fhe.q[level] / float(1 << fhe.log_default_scale)
            correct_square_scale = not deferred_rescale and (rescale_ratio > 2.0 or rescale_ratio < 0.5)

            c0_level = level if deferred_rescale else level - 1
            c0_pt = [CkksPlaintextNode(f'herpnw_{layer_id}_0_{i}', level=c0_level) for i in range(len(input_nodes))]
            c1_pt = [CkksPlaintextMulNode(f'herpnw_{layer_id}_1_{i}', level=level) for i in range(len(input_nodes))]
            square_scale_pt = (
                [CkksPlaintextMulNode(f'herpnw_{layer_id}_2_{i}', level=level) for i in range(len(input_nodes))]
                if correct_square_scale else []
            )

            layer_output_nodes = []
            for i, x_node in enumerate(input_nodes):
                square = relin(mult(x_node, x_node))
                if correct_square_scale:
                    square = mult(square, square_scale_pt[i])
                linear = mult(x_node, c1_pt[i])
                if not deferred_rescale:
                    square = rescale(square)
                    linear = rescale(linear)
                out = add(square, linear)
                out = add(out, c0_pt[i])
                layer_output_nodes.append(out)
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
            input_args.append(Argument(f'herpnw_{layer_id}_0', c0_pt))
            input_args.append(Argument(f'herpnw_{layer_id}_1', c1_pt))
            if correct_square_scale:
                input_args.append(Argument(f'herpnw_{layer_id}_2', square_scale_pt))

        elif layer_config['type'] == 'conv1d':
            input_shape = config_info['feature'][layer_input_feature_ids[0]]['shape'][0]
            kernel_shape = layer_config['kernel_shape'][0]
            stride = layer_config['stride'][0]
            skip_1d = skip[0] if isinstance(skip, list) else skip
            n_channel_per_ct = int(n // 2 // input_shape // skip_1d)
            n_pack_in_channel = math.ceil(n_in_channel / n_channel_per_ct)
            n_packed_out_channel = math.ceil(n_out_channel / (n_channel_per_ct * stride))
            rot_n_channel_per_ct = min(n_in_channel, n_channel_per_ct)
            input_ct = [
                CkksCiphertextNode(f'{layer_input_feature_ids[0]}input{k}', level=level)
                for k in range(n_pack_in_channel)
            ]
            if layer_input_feature_ids[0] not in feature_id_to_nodes_map:
                feature_id_to_nodes_map[layer_input_feature_ids[0]] = input_ct
                input_args.append(Argument(layer_input_feature_ids[0], input_ct))
            weight_pt = [
                [
                    [CkksPlaintextRingtNode(f'convw_{layer_id}_{i}_{k}_{j}') for k in range(kernel_shape)]
                    for j in range(rot_n_channel_per_ct)
                ]
                for i in range(n_packed_out_channel)
            ]
            bias_pt = [CkksPlaintextRingtNode(f'convb_{layer_id}_{i}') for i in range(n_packed_out_channel)]
            conv1d = Conv1DPackedLayer(
                n_out_channel,
                n_in_channel,
                input_shape,
                kernel_shape,
                stride,
                skip_1d,
                n_channel_per_ct,
                n_pack_in_channel,
                n_packed_out_channel,
            )
            layer_output_nodes = conv1d.call(feature_id_to_nodes_map[layer_input_feature_ids[0]], weight_pt, bias_pt)
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
            input_args.append(Argument(f'convw_{layer_id}', weight_pt))
            input_args.append(Argument(f'convb_{layer_id}', bias_pt))

        elif 'mult_scalar' in layer_config['type']:
            mult_scalar_layer = MultScalarLayer()
            input_nodes = feature_id_to_nodes_map[layer_input_feature_ids[0]]
            pt = [CkksPlaintextRingtNode(f'mult_scalar_{layer_id}_{i}') for i in range(len(input_nodes))]
            input_level = config_info['feature'][layer_input_feature_ids[0]]['level']
            output_level = config_info['feature'][layer_output_feature_ids[0]]['level']
            layer_output_nodes = mult_scalar_layer.call(
                input_nodes, pt, rescale_output=output_level < input_level
            )
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
            input_args.append(Argument(f'mult_scalar_{layer_id}', pt))

        elif layer_config['type'] == 'mult_coeff':
            raise ValueError(
                f"Layer '{layer_id}' has type 'mult_coeff', which should have been absorbed "
                f"into adjacent layers or converted to 'mult_scalar' during compilation."
            )

        elif layer_config['type'] == 'drop_level':
            level_in = config_info['feature'][layer_input_feature_ids[0]]['level']
            level_out = config_info['feature'][layer_output_feature_ids[0]]['level']
            drop_level_n = level_in - level_out
            input_nodes = feature_id_to_nodes_map[layer_input_feature_ids[0]]
            if drop_level_n <= 0:
                layer_output_nodes = input_nodes
            else:
                layer_output_nodes = [drop_level(node, drop_level_n) for node in input_nodes]
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif layer_config['type'] == 'bootstrapping':
            layer_output_nodes = []
            for node in feature_id_to_nodes_map[layer_input_feature_ids[0]]:
                # Keep one residual modulus so the backend can match the
                # bootstrap input scale before dropping to level zero.
                if node.level > 1:
                    node = drop_level(node, node.level - 1)
                layer_output_nodes.append(bootstrap(node))
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif layer_config['type'] == 'l2_norm_poly':
            input_nodes = feature_id_to_nodes_map[layer_input_feature_ids[0]]
            input_feature = config_info['feature'][layer_input_feature_ids[0]]
            output_feature = config_info['feature'][layer_output_feature_ids[0]]
            skip_0d = int(input_feature['skip'])
            channels_per_ct = int(input_feature['pack_num'])
            target_level = int(output_feature['level'])

            input_level = int(input_feature['level'])
            # 4-level implementation: a·y² + b·y + c, matching the orion L2NormPoly.
            # ct-ct rescales: x*x (L-1), y*y (L-2). pt-ct rescales: a (L-3), b (L-3).
            # poly at L-3, scale Δ. Final x*poly + rescale → L-4, scale Δ (standard output).
            coeff_a = [CkksPlaintextMulNode(f'l2norm_{layer_id}_a_0', level=input_level - 2)]
            coeff_b = [CkksPlaintextMulNode(f'l2norm_{layer_id}_b_0', level=input_level - 2)]
            coeff_c = [CkksPlaintextNode(f'l2norm_{layer_id}_c_0', level=input_level - 3)]

            # Step 1: x_sq and y  (1 rescale)
            partial_sums = []
            for x_node in input_nodes:
                x_sq = rescale(relin(mult(x_node, x_node)))
                sum_node = x_sq
                for i in range(int(math.ceil(math.log2(channels_per_ct)))):
                    sum_node = add(sum_node, rotate_cols(sum_node, [(1 << i) * skip_0d])[0])
                partial_sums.append(sum_node)

            norm_sq = partial_sums[0]
            for partial in partial_sums[1:]:
                if norm_sq.level > partial.level:
                    norm_sq = drop_level(norm_sq, norm_sq.level - partial.level)
                elif partial.level > norm_sq.level:
                    partial = drop_level(partial, partial.level - norm_sq.level)
                norm_sq = add(norm_sq, partial)

            # Step 2: y²  (1 rescale)
            norm_sq_sq = rescale(relin(mult(norm_sq, norm_sq)))

            # Align y down to match y² level
            norm_sq_for_b = norm_sq
            if norm_sq_for_b.level > norm_sq_sq.level:
                norm_sq_for_b = drop_level(norm_sq_for_b, norm_sq_for_b.level - norm_sq_sq.level)

            # Step 3: poly = a·y² + b·y + c  (rescale after each pt-ct mult → scale Δ)
            # norm_sq_sq and norm_sq_for_b are both at L-2; a/b pt_mul are encoded at L-2.
            a_term = rescale(mult(norm_sq_sq, coeff_a[0]))
            b_term = rescale(mult(norm_sq_for_b, coeff_b[0]))
            inv_norm = add(a_term, b_term)
            inv_norm = add(inv_norm, coeff_c[0])

            # Step 4: x · poly  (1 rescale)
            layer_output_nodes = []
            for x_node in input_nodes:
                x_for_mul = x_node
                norm_for_mul = inv_norm
                if x_for_mul.level > inv_norm.level:
                    x_for_mul = drop_level(x_for_mul, x_for_mul.level - inv_norm.level)
                elif inv_norm.level > x_for_mul.level:
                    norm_for_mul = drop_level(inv_norm, inv_norm.level - x_for_mul.level)
                y_node = rescale(relin(mult(x_for_mul, norm_for_mul)))
                if y_node.level > target_level:
                    y_node = drop_level(y_node, y_node.level - target_level)
                layer_output_nodes.append(y_node)

            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
            input_args.append(Argument(f'l2norm_{layer_id}_a', coeff_a))
            input_args.append(Argument(f'l2norm_{layer_id}_b', coeff_b))
            input_args.append(Argument(f'l2norm_{layer_id}_c', coeff_c))

        elif layer_config['type'] in ('add', 'add2d'):
            layer_output_nodes = []
            lhs_nodes = feature_id_to_nodes_map[layer_input_feature_ids[0]]
            rhs_nodes = feature_id_to_nodes_map[layer_input_feature_ids[1]]
            for lhs, rhs in zip(lhs_nodes, rhs_nodes):
                if lhs.level > rhs.level:
                    lhs = drop_level(lhs, lhs.level - rhs.level)
                elif rhs.level > lhs.level:
                    rhs = drop_level(rhs, rhs.level - lhs.level)
                layer_output_nodes.append(add(lhs, rhs))
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif 'concat2d' in layer_config['type']:
            # Check if any input has n_channel not divisible by n_channel_per_ct
            input_n_channels = []
            has_uneven = False
            for input_fid in layer_input_feature_ids:
                feat = config_info['feature'][input_fid]
                n_ch = int(feat['channel'])
                input_n_channels.append(n_ch)
                if n_ch % pack != 0:
                    has_uneven = True

            if has_uneven:
                # Uneven path: per-channel mask+rotate+add
                concat_layer = ConcatLayer()
                input_node_lists = [feature_id_to_nodes_map[fid] for fid in layer_input_feature_ids]
                input_shape = config_info['feature'][layer_input_feature_ids[0]]['shape']
                total_channels = sum(input_n_channels)

                # Create mask plaintext nodes for each global channel
                mask_pts = [CkksPlaintextRingtNode(f'concat_mask_{layer_id}_{i}') for i in range(total_channels)]

                layer_output_nodes = concat_layer.call_multiple_inputs_uneven(
                    input_node_lists, input_n_channels, pack, input_shape, skip, mask_pts
                )
                feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
                input_args.append(Argument(f'concat_mask_{layer_id}', mask_pts))
            else:
                # Fast path: concat is a runtime-only operation, just merge node lists
                layer_output_nodes = []
                for input_fid in layer_input_feature_ids:
                    layer_output_nodes.extend(feature_id_to_nodes_map[input_fid])
                feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif 'upsample_nearest' in layer_config['type']:
            input_shape = config_info['feature'][layer_input_feature_ids[0]]['shape']
            upsample_factor = layer_config['upsample_factor']
            n_channel_per_ct = pack
            upsample_layer = UpsampleNearestLayer(
                shape=input_shape,
                skip=skip,
                upsample_factor=upsample_factor,
                n_channel_per_ct=n_channel_per_ct,
                level=level,
            )
            out_channels_per_ct = n_channel_per_ct // (upsample_factor[0] * upsample_factor[1])
            n_select_pt = out_channels_per_ct
            select_tensor_pt = [
                CkksPlaintextRingtNode(f'upsample_select_pt_{layer_id}_{i}') for i in range(n_select_pt)
            ]
            layer_output_nodes = upsample_layer.call(
                feature_id_to_nodes_map[layer_input_feature_ids[0]],
                select_tensor_pt,
                n_in_channel,
            )
            input_args.append(Argument(f'upsample_select_pt_{layer_id}', select_tensor_pt))
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif 'avgpool' in layer_config['type']:
            input_shape = config_info['feature'][layer_input_feature_ids[0]]['shape']
            stride = layer_config['stride']
            is_adaptive = layer_config.get('is_adaptive_avgpool', True)
            is_big_size = layer_config.get('is_big_size', False)
            avgpool = Avgpool2DLayer(stride, input_shape, channel=n_in_channel, skip=skip)
            if is_big_size:
                block_expansion = [
                    math.ceil(input_shape[0] / block_shape[0]),
                    math.ceil(input_shape[1] / block_shape[1]),
                ]
                layer_output_nodes = avgpool.call_interleaved_avgpool(
                    feature_id_to_nodes_map[layer_input_feature_ids[0]], block_expansion, N=n
                )
            else:
                if is_adaptive:
                    # level_cost=0: only rotations + adds, normalization absorbed into adjacent layers
                    if style == 'ordinary':
                        layer_output_nodes = avgpool.call(feature_id_to_nodes_map[layer_input_feature_ids[0]])
                    else:
                        layer_output_nodes = avgpool.run_adaptive_avgpool(
                            feature_id_to_nodes_map[layer_input_feature_ids[0]], n=n
                        )
                else:
                    # level_cost=1: non-adaptive avgpool needs mult+rescale (select_tensor)
                    n_channel_per_ct = int(math.ceil(n / 2 / (input_shape[0] * input_shape[1])))
                    out_channels_per_ct = n_channel_per_ct * stride[0] * stride[1]
                    n_select_pt = min(n_in_channel, out_channels_per_ct)
                    select_tensor_pt = [CkksPlaintextRingtNode(f'select_pt_{layer_id}_{i}') for i in range(n_select_pt)]
                    layer_output_nodes = avgpool.call_multiplexed_avgpool(
                        feature_id_to_nodes_map[layer_input_feature_ids[0]],
                        select_tensor_pt,
                        n_in_channel,
                        n_channel_per_ct,
                    )
                    input_args.append(Argument(f'select_tensor_pt_{layer_id}', select_tensor_pt))

            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif layer_config['type'] == 'fc0':
            n_packed_in_channel = math.ceil(n_in_channel / (n // 2))
            n_packed_out_channel = math.ceil(n_out_channel / pack)
            if 'special_info' not in config_info['feature'][layer_input_feature_ids[0]]:
                # call_skip_0d path — matching test_fc_fc_feature0d Layer 1
                skip_0d = config_info['feature'][layer_input_feature_ids[0]]['skip']
                n_channel_per_ct = int(n // 2 // skip_0d)
                pack_0d = n_channel_per_ct
                n_packed_in_feature = math.ceil(n_in_channel / n_channel_per_ct)
                n_packed_out_feature = math.ceil(n_out_channel / n_channel_per_ct)
                weight_pt_size = n_packed_in_feature * pack_0d
                weight_pt = [
                    [CkksPlaintextRingtNode(f'densew_{layer_id}_{m}_{i}') for i in range(weight_pt_size)]
                    for m in range(n_packed_out_feature)
                ]
                bias_pt = [CkksPlaintextRingtNode(f'denseb_{layer_id}_{i}') for i in range(n_packed_out_feature)]
                fc_layer = DensePackedLayer(
                    n_out_channel,
                    n_in_channel,
                    [1, 1],
                    [1, 1],
                    pack_0d,
                    n_packed_in_feature,
                    n_packed_out_feature,
                )
                input_args.append(Argument(f'densew_{layer_id}', weight_pt))
                input_args.append(Argument(f'denseb_{layer_id}', bias_pt))
                layer_output_nodes = fc_layer.call_skip_0d(
                    feature_id_to_nodes_map[layer_input_feature_ids[0]], weight_pt, bias_pt, skip_0d
                )
            else:
                special_info = config_info['feature'][layer_input_feature_ids[0]]['special_info']
                special_shape = special_info['shape']
                special_skip = special_info['skip']
                invalid_fill = special_info.get('invalid_fill', [1, 1])
                input_shape_ct = [special_shape[0] * special_skip[0], special_shape[1] * special_skip[1]]
                n_num_per_ct = math.ceil(n // 2 / (input_shape_ct[0] * input_shape_ct[1]))
                n_packed_out_feature_for_mult_apck = math.ceil(n_out_channel / n_num_per_ct)
                valid_skip_0 = special_skip[0] // invalid_fill[0]
                valid_skip_1 = special_skip[1] // invalid_fill[1]
                n_channel_per_block = valid_skip_0 * valid_skip_1
                n_channel = n_in_channel // (special_shape[0] * special_shape[1])
                n_block_input = math.ceil(n_channel / (n_num_per_ct * n_channel_per_block)) * n_num_per_ct
                weight_pt = [
                    [CkksPlaintextRingtNode(f'densew_{layer_id}_{i}_{j}') for j in range(n_block_input)]
                    for i in range(n_packed_out_feature_for_mult_apck)
                ]
                bias_pt = [
                    CkksPlaintextRingtNode(f'denseb_{layer_id}_{i}') for i in range(n_packed_out_feature_for_mult_apck)
                ]
                dense = DensePackedLayer(
                    n_out_channel,
                    n_in_channel,
                    special_shape,
                    special_skip,
                    n_num_per_ct,
                    n_in_channel,
                    n_out_channel,
                    invalid_fill=invalid_fill,
                )
                input_args.append(Argument(f'densew_{layer_id}', weight_pt))
                input_args.append(Argument(f'denseb_{layer_id}', bias_pt))
                layer_output_nodes = dense.call_multiplexed(
                    feature_id_to_nodes_map[layer_input_feature_ids[0]], weight_pt, bias_pt, n
                )
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        else:
            raise ValueError(f'Unsupported layer type: {layer_config["type"]}')

    output_args = [Argument(output_id, feature_id_to_nodes_map[output_id]) for output_id in task_output_feature_ids]

    process_custom_task(input_args=input_args, output_args=output_args, output_instruction_path=task_path, fpga_acc=False)


if __name__ == '__main__':
    if hasattr(sys, 'frozen'):
        os.chdir(os.path.dirname(sys.executable))
    else:
        os.chdir(os.path.dirname(__file__))

    parser = argparse.ArgumentParser(description='FPGA config generator.')
    parser.add_argument('task_path', type=str, help='Path of the server directory')
    args = parser.parse_args()

    task_path = args.task_path
    with open(os.path.join(task_path, 'task_config.json'), 'r', encoding='utf-8') as file:
        config = json.load(file)

    for _, is_fpga in config['server_task'].items():
        if is_fpga['enable_fpga']:
            server_path = os.path.join(task_path, 'server')
            # Read poly_modulus_degree from ckks_parameter.json to set correct N
            ckks_param_path = os.path.join(server_path, 'ckks_parameter.json')
            poly_n = 16384
            if os.path.exists(ckks_param_path):
                with open(ckks_param_path, 'r', encoding='utf-8') as f:
                    ckks_cfg = json.load(f)
                # Get first param entry's poly_modulus_degree
                for param_id, param_cfg in ckks_cfg.items():
                    poly_n = param_cfg.get('poly_modulus_degree', 16384)
                    break
            gen_custom_task(server_path, n=poly_n, use_gpu=True)
