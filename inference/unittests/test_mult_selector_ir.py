# Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Rung 1: IR construction unit test for mult_selector.

Verifies the IR-level contract:
- Output is a CkksCiphertextNode at ct.level.
- Op type serializes to 'mult_selector'.
- Non-ringt mask is rejected with a clear error.

Path note: custom_task.py uses bare ``from frontend.bootstrap_params import``
so inference/lattisense must be on sys.path. We add it explicitly before
importing the module (the lattisense tests in inference/lattisense/unittests/
do the same via project_root = dirname(dirname(__file__))).

reset_dag() does not exist in custom_task.py; DAG is reset manually by
reassigning the module-level g_dag to a fresh nx.DiGraph() (approach b).
"""

import unittest
from pathlib import Path
import sys

import networkx as nx

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
# lattisense must be on sys.path so 'from frontend.bootstrap_params import'
# resolves — the submodule uses bare package names relative to its own root.
lattisense_root = project_root / 'inference' / 'lattisense'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(lattisense_root))

import inference.lattisense.frontend.custom_task as ct_mod
from inference.lattisense.frontend.custom_task import (
    CkksCiphertextNode,
    CkksPlaintextRingtNode,
    CkksPlaintextNode,
    OperationType,
    mult_selector,
)


def _reset_dag():
    """Reset the module-level DAG and random-id set between tests."""
    ct_mod.g_dag = nx.DiGraph()
    ct_mod.random_ids = set()


class TestMultSelectorIR(unittest.TestCase):
    def setUp(self):
        _reset_dag()
        self.ct = CkksCiphertextNode(id='ct_in', level=4)
        self.mask = CkksPlaintextRingtNode(id='mask_in')
        self.mask.level = 4

    def test_output_level_preserved(self):
        out = mult_selector(self.ct, self.mask, output_id='out')
        self.assertIsInstance(out, CkksCiphertextNode)
        self.assertEqual(out.level, self.ct.level)
        self.assertEqual(out.id, 'out')

    def test_op_type_serialization(self):
        self.assertEqual(OperationType.MultSelector.value, 'mult_selector')

    def test_rejects_non_ringt_mask(self):
        # setUp already reset the DAG; reuse self.ct and build only the bad mask.
        bad_mask = CkksPlaintextNode(id='bad_mask')
        bad_mask.level = self.ct.level
        with self.assertRaises(TypeError):
            mult_selector(self.ct, bad_mask)

    # Scale preservation is part of the runtime contract (ct.scale × pt.scale
    # with pt.scale=1.0 → output.scale=ct.scale) but CkksCiphertextNode does
    # not carry a scale attribute at the IR layer, so scale invariance is
    # asserted in the Rung 2 C++ micro-test, not here.


if __name__ == '__main__':
    unittest.main()
