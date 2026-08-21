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

import logging

from typing_extensions import override
from . import ComputeNode, FeatureNode, format_id, dict_to_args
from onnx import NodeProto

log = logging.getLogger(__name__)


class HerPNComputeNode(ComputeNode):
    """CryptoFace factored HerPN quadratic: x^2 + c1*x + c0."""

    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        feature_input: list[FeatureNode],
        feature_output: list[FeatureNode],
        module_path: str = '',
    ):
        super(HerPNComputeNode, self).__init__(layer_id, layer_type, feature_input, feature_output)
        self.layer_type = 'herpn2d'
        self.module_path = module_path
        feature_output[0].channel = feature_input[0].channel
        feature_output[0].level = feature_input[0].level - 1
        feature_output[0].shape = feature_input[0].shape
        feature_output[0].scale = feature_input[0].scale
        feature_output[0].skip = feature_input[0].skip

    @staticmethod
    def from_onnx_node(x: NodeProto, features_nodes) -> 'HerPNComputeNode':
        layer_id = format_id(x.name)
        feature_input = [features_nodes[format_id(x.input[0])]]
        feature_output = [features_nodes[format_id(x.output[0])]]

        module_path = ''
        parts = x.name.strip('/').split('/')
        if len(parts) >= 2:
            module_path = '.'.join(parts[:-1])
        else:
            module_path = layer_id

        return HerPNComputeNode(layer_id, 'herpn2d', feature_input, feature_output, module_path)

    @override
    def to_torch_code(self) -> dict[str, list[str]]:
        log.debug('Generating herpn2d code')
        init_str, forward_str = [], []
        params = dict_to_args({'num_features': self.feature_input[0].channel})
        init_str.append(f'self.{self.layer_id} = HerPNComputeNode({params})')
        return {'init': init_str, 'forward': forward_str}

    @override
    def to_json(self) -> dict:
        info = dict()
        info['type'] = self.layer_type
        info['channel_input'] = self.feature_input[0].channel
        info['channel_output'] = self.feature_output[0].channel
        info['ckks_parameter_id_input'] = self.feature_input[0].ckks_parameter_id
        info['ckks_parameter_id_output'] = self.feature_output[0].ckks_parameter_id
        info['feature_input'] = [i.node_id for i in self.feature_input]
        info['feature_output'] = [i.node_id for i in self.feature_output]
        info['weight_path'] = f'{self.module_path}.weight'
        return info
